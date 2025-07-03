import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils import *

class TimeSeriesDaGMM(nn.Module):
    """Time Series Deep Autoencoding Gaussian Mixture Model for Anomaly Detection."""
    
    def __init__(self, 
                 input_dim=256, 
                 latent_dim=32, 
                 n_gmm=4, 
                 hidden_dim=64, 
                 num_layers=2, 
                 dropout=0.2,
                 bidirectional=True):
        super(TimeSeriesDaGMM, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.n_gmm = n_gmm
        
        # LSTM Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Encoder projection to latent space
        encoder_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.encoder_projection = nn.Sequential(
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # LSTM Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Decoder is unidirectional
        )
        
        # Decoder projection to output space
        self.decoder_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Feature dimension for GMM: latent_dim + temporal reconstruction features
        feature_dim = latent_dim + 4  # latent + mse + mae + dtw_like + temporal_trend
        
        # GMM estimation network
        self.estimation = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(32, n_gmm),
            nn.Softmax(dim=1)
        )
        
        # GMM parameters
        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, feature_dim))
        self.register_buffer("cov", torch.zeros(n_gmm, feature_dim, feature_dim))
        
    def encode(self, x, mask=None):
        """
        Encode time series to latent representation.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Mask tensor [batch_size, seq_len] for variable length sequences
        
        Returns:
            latent: Latent representation [batch_size, latent_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.encoder_lstm(x)
        
        if mask is not None:
            # Use mask to get the last valid timestep for each sequence
            mask_expanded = mask.unsqueeze(-1).expand_as(lstm_out)
            lstm_out = lstm_out * mask_expanded
            
            # Get the last valid output for each sequence
            lengths = mask.sum(dim=1) - 1  # -1 because of 0-indexing
            batch_indices = torch.arange(batch_size, device=x.device)
            last_outputs = lstm_out[batch_indices, lengths.long()]
        else:
            # Use the last timestep output
            if self.bidirectional:
                # For bidirectional LSTM, concatenate forward and backward hidden states
                last_outputs = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                last_outputs = h_n[-1]
        
        # Project to latent space
        latent = self.encoder_projection(last_outputs)
        return latent
    
    def decode(self, latent, target_seq_len):
        """
        Decode latent representation to time series.
        
        Args:
            latent: Latent representation [batch_size, latent_dim]
            target_seq_len: Target sequence length for reconstruction
            
        Returns:
            decoded: Reconstructed time series [batch_size, target_seq_len, input_dim]
        """
        batch_size = latent.size(0)
        
        # Repeat latent representation for each timestep
        decoder_input = latent.unsqueeze(1).repeat(1, target_seq_len, 1)
        
        # LSTM decoding
        lstm_out, _ = self.decoder_lstm(decoder_input)
        
        # Project to output space
        decoded = self.decoder_projection(lstm_out)
        return decoded
    
    def compute_temporal_features(self, x, x_hat, mask=None):
        """
        Compute time series specific reconstruction features.
        
        Args:
            x: Original time series [batch_size, seq_len, input_dim]
            x_hat: Reconstructed time series [batch_size, seq_len, input_dim]
            mask: Mask tensor [batch_size, seq_len]
            
        Returns:
            features: Temporal features [batch_size, 4]
        """
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x_masked = x * mask_expanded
            x_hat_masked = x_hat * mask_expanded
            lengths = mask.sum(dim=1, keepdim=True).float()
        else:
            x_masked = x
            x_hat_masked = x_hat
            lengths = torch.full((x.size(0), 1), x.size(1), device=x.device, dtype=torch.float)
        
        # Mean Squared Error per sequence
        mse = torch.mean((x_masked - x_hat_masked) ** 2, dim=(1, 2))
        
        # Mean Absolute Error per sequence
        mae = torch.mean(torch.abs(x_masked - x_hat_masked), dim=(1, 2))
        
        # DTW-like distance (simplified version)
        diff = torch.abs(x_masked - x_hat_masked)
        if mask is not None:
            dtw_like = torch.sum(diff, dim=(1, 2)) / lengths.squeeze()
        else:
            dtw_like = torch.mean(diff, dim=(1, 2))
        
        # Temporal trend difference
        x_diff = x_masked[:, 1:] - x_masked[:, :-1]
        x_hat_diff = x_hat_masked[:, 1:] - x_hat_masked[:, :-1]
        trend_diff = torch.mean(torch.abs(x_diff - x_hat_diff), dim=(1, 2))
        
        features = torch.stack([mse, mae, dtw_like, trend_diff], dim=1)
        return features
    
    def forward(self, x, mask=None):
        """
        Forward pass of the Time Series DAGMM.
        
        Args:
            x: Input time series [batch_size, seq_len, input_dim]
            mask: Mask tensor [batch_size, seq_len] for variable length sequences
            
        Returns:
            latent: Latent representation
            decoded: Reconstructed time series
            z: Feature vector for GMM
            gamma: GMM component probabilities
        """
        batch_size, seq_len, _ = x.size()
        
        # Encode
        latent = self.encode(x, mask)
        
        # Decode
        decoded = self.decode(latent, seq_len)
        
        # Compute temporal features
        temporal_features = self.compute_temporal_features(x, decoded, mask)
        
        # Combine latent representation with temporal features
        z = torch.cat([latent, temporal_features], dim=1)
        
        # Estimate GMM component probabilities
        gamma = self.estimation(z)
        
        return latent, decoded, z, gamma
    
    def compute_gmm_params(self, z, gamma):
        """
        Compute GMM parameters from feature vectors and component probabilities.
        
        Args:
            z: Feature vectors [batch_size, feature_dim]
            gamma: Component probabilities [batch_size, n_gmm]
            
        Returns:
            phi: Component weights [n_gmm]
            mu: Component means [n_gmm, feature_dim]
            cov: Component covariances [n_gmm, feature_dim, feature_dim]
        """
        N = gamma.size(0)
        
        # Component weights
        sum_gamma = torch.sum(gamma, dim=0)
        phi = sum_gamma / N
        self.phi = phi.data
        
        # Component means
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        
        # Component covariances
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)  # [N, n_gmm, feature_dim]
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)  # [N, n_gmm, feature_dim, feature_dim]
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data
        
        return phi, mu, cov
    
    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        """
        Compute sample energy for anomaly detection.
        
        Args:
            z: Feature vectors [batch_size, feature_dim]
            phi: Component weights [n_gmm]
            mu: Component means [n_gmm, feature_dim]
            cov: Component covariances [n_gmm, feature_dim, feature_dim]
            size_average: Whether to average over batch
            
        Returns:
            sample_energy: Energy values
            cov_diag: Covariance diagonal regularization term
        """
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)
        
        k, D, _ = cov.size()
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)  # [N, k, D]
        
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        
        for i in range(k):
            cov_k = cov[i] + to_var(torch.eye(D) * eps)
            try:
                cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
                det_cov.append(torch.det(cov_k).unsqueeze(0))
            except RuntimeError:
                # Handle singular matrices
                U, S, V = torch.svd(cov_k)
                S_inv = torch.where(S > eps, 1.0 / S, torch.zeros_like(S))
                cov_inv = torch.mm(torch.mm(V, torch.diag(S_inv)), U.t())
                cov_inverse.append(cov_inv.unsqueeze(0))
                det_val = torch.prod(torch.clamp(S, min=eps))
                det_cov.append(det_val.unsqueeze(0))
            
            cov_diag = cov_diag + torch.sum(1 / torch.clamp(cov_k.diag(), min=eps))
        
        cov_inverse = torch.cat(cov_inverse, dim=0)  # [k, D, D]
        det_cov = torch.cat(det_cov)  # [k]
        
        # Compute energy
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        max_val = torch.max(exp_term_tmp.clamp(min=0), dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)
        
        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / torch.sqrt(torch.clamp(det_cov * (2 * np.pi) ** D, min=eps)).unsqueeze(0), dim=1) + eps
        )
        
        if size_average:
            sample_energy = torch.mean(sample_energy)
        
        return sample_energy, cov_diag
    
    def loss_function(self, x, x_hat, z, gamma, mask=None, lambda_energy=0.1, lambda_cov_diag=0.005):
        """
        Compute loss function for training.
        
        Args:
            x: Original time series [batch_size, seq_len, input_dim]
            x_hat: Reconstructed time series [batch_size, seq_len, input_dim]
            z: Feature vectors [batch_size, feature_dim]
            gamma: Component probabilities [batch_size, n_gmm]
            mask: Mask tensor [batch_size, seq_len]
            lambda_energy: Weight for energy term
            lambda_cov_diag: Weight for covariance diagonal regularization
            
        Returns:
            loss: Total loss
            sample_energy: Energy term
            recon_error: Reconstruction error
            cov_diag: Covariance diagonal term
        """
        # Compute reconstruction error with mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            masked_diff = (x - x_hat) * mask_expanded
            recon_error = torch.sum(masked_diff ** 2) / torch.sum(mask_expanded)
        else:
            recon_error = torch.mean((x - x_hat) ** 2)
        
        # Compute GMM parameters and energy
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        
        # Total loss
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        
        return loss, sample_energy, recon_error, cov_diag