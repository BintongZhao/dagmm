import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils import *

class Cholesky(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        l = torch.linalg.cholesky(a)
        ctx.save_for_backward(l)
        return l
    
    @staticmethod
    def backward(ctx, grad_output):
        l, = ctx.saved_tensors
        linv = torch.inverse(l)
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - torch.eye(l.size(1), device=l.device, dtype=l.dtype).fill_(0.5))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s

class DaGMM_TimeSeries(nn.Module):
    """Time Series Deep Autoencoding Gaussian Mixture Model for Anomaly Detection."""
    
    def __init__(self, input_dim=256, sequence_length=None, latent_dim=8, hidden_dim=64, n_gmm=2):
        super(DaGMM_TimeSeries, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_gmm = n_gmm
        
        # Encoder: LSTM to process temporal sequence
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Encoder output layer to get latent representation
        self.encoder_output = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )
        
        # Decoder: from latent representation back to sequence
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.decoder_output = nn.Linear(hidden_dim, input_dim)
        
        # Estimation network for GMM (similar to original)
        # Input: latent_dim + 2 (reconstruction errors) = latent_dim + 2
        estimation_input_dim = latent_dim + 2
        
        layers = []
        layers += [nn.Linear(estimation_input_dim, 10)]
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=0.5)]
        layers += [nn.Linear(10, n_gmm)]
        layers += [nn.Softmax(dim=1)]
        
        self.estimation = nn.Sequential(*layers)
        
        # Register GMM parameters
        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, estimation_input_dim))
        self.register_buffer("cov", torch.zeros(n_gmm, estimation_input_dim, estimation_input_dim))
    
    def encode(self, x):
        """
        Encode time series to latent representation
        Args:
            x: [batch_size, sequence_length, input_dim]
        Returns:
            latent: [batch_size, latent_dim]
        """
        batch_size = x.size(0)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.encoder_lstm(x)
        
        # Use the last hidden state for latent representation
        # Take the last layer's hidden state
        latent = self.encoder_output(hidden[-1])  # [batch_size, latent_dim]
        
        return latent
    
    def decode(self, latent, sequence_length):
        """
        Decode latent representation back to time series
        Args:
            latent: [batch_size, latent_dim]
            sequence_length: length of sequence to generate
        Returns:
            reconstructed: [batch_size, sequence_length, input_dim]
        """
        batch_size = latent.size(0)
        
        # Transform latent to hidden dimension
        hidden_input = self.decoder_input(latent)  # [batch_size, hidden_dim]
        
        # Prepare input for LSTM decoder
        # Repeat the hidden input for each time step
        decoder_input = hidden_input.unsqueeze(1).repeat(1, sequence_length, 1)
        
        # LSTM decoding
        lstm_out, _ = self.decoder_lstm(decoder_input)
        
        # Output layer
        reconstructed = self.decoder_output(lstm_out)  # [batch_size, seq_len, input_dim]
        
        return reconstructed
    
    def relative_euclidean_distance(self, a, b):
        """
        Compute relative euclidean distance between original and reconstructed sequences
        Args:
            a, b: [batch_size, sequence_length, input_dim]
        Returns:
            distance: [batch_size]
        """
        # Flatten the sequences for distance computation
        a_flat = a.view(a.size(0), -1)  # [batch_size, seq_len * input_dim]
        b_flat = b.view(b.size(0), -1)  # [batch_size, seq_len * input_dim]
        
        return (a_flat - b_flat).norm(2, dim=1) / (a_flat.norm(2, dim=1) + 1e-8)
    
    def cosine_similarity_sequence(self, a, b):
        """
        Compute cosine similarity between original and reconstructed sequences
        Args:
            a, b: [batch_size, sequence_length, input_dim]
        Returns:
            similarity: [batch_size]
        """
        # Flatten the sequences for similarity computation
        a_flat = a.view(a.size(0), -1)  # [batch_size, seq_len * input_dim]
        b_flat = b.view(b.size(0), -1)  # [batch_size, seq_len * input_dim]
        
        return F.cosine_similarity(a_flat, b_flat, dim=1)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: [batch_size, sequence_length, input_dim]
        Returns:
            enc: latent representation
            dec: reconstructed sequence
            z: augmented latent representation (enc + reconstruction errors)
            gamma: GMM component weights
        """
        batch_size, sequence_length, input_dim = x.size()
        
        # Encode
        enc = self.encode(x)  # [batch_size, latent_dim]
        
        # Decode
        dec = self.decode(enc, sequence_length)  # [batch_size, seq_len, input_dim]
        
        # Compute reconstruction errors
        rec_cosine = self.cosine_similarity_sequence(x, dec)  # [batch_size]
        rec_euclidean = self.relative_euclidean_distance(x, dec)  # [batch_size]
        
        # Augment latent representation with reconstruction errors
        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
        
        # Compute GMM component weights
        gamma = self.estimation(z)
        
        return enc, dec, z, gamma
    
    def compute_gmm_params(self, z, gamma):
        """
        Compute GMM parameters (same as original DAGMM)
        """
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)
        
        # K
        phi = (sum_gamma / N)
        self.phi = phi.data
        
        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        
        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        
        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        
        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data
        
        return phi, mu, cov
    
    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        """
        Compute sample energy (same as original DAGMM)
        """
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)
        
        k, D, _ = cov.size()
        
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-8  # Increased regularization
        
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(D) * eps)
            
            # Add diagonal loading to ensure positive definiteness
            cov_k = cov_k + to_var(torch.eye(D) * eps * 10)
            
            try:
                cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
                # Use log determinant for numerical stability
                det_cov.append(torch.det(cov_k * (2 * np.pi)).unsqueeze(0))
            except:
                # Fallback: use identity if matrix is still not invertible
                cov_k = to_var(torch.eye(D))
                cov_inverse.append(cov_k.unsqueeze(0))
                det_cov.append(torch.det(cov_k * (2 * np.pi)).unsqueeze(0))
            
            cov_diag = cov_diag + torch.sum(1 / (cov_k.diag() + eps))
        
        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = torch.cat(det_cov)
        if torch.cuda.is_available():
            det_cov = det_cov.cuda()
        
        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]
        
        exp_term = torch.exp(exp_term_tmp - max_val)
        
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov + eps)).unsqueeze(0), dim=1) + eps)
        
        if size_average:
            sample_energy = torch.mean(sample_energy)
        
        return sample_energy, cov_diag
    
    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        """
        Compute total loss (same structure as original DAGMM)
        """
        # Compute reconstruction error for time series
        recon_error = torch.mean((x - x_hat) ** 2)
        
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        
        return loss, sample_energy, recon_error, cov_diag