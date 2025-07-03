import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import Variable
from model_timeseries import TimeSeriesDaGMM
from data_loader_timeseries import get_timeseries_loader, collate_fn
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score, roc_auc_score
import logging

class TimeSeriesSolver(object):
    """Solver for Time Series DAGMM training and testing."""
    
    DEFAULTS = {
        'lr': 1e-4,
        'num_epochs': 100,
        'batch_size': 32,
        'input_dim': 256,
        'latent_dim': 32,
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'bidirectional': True,
        'n_gmm': 4,
        'lambda_energy': 0.1,
        'lambda_cov_diag': 0.005,
        'log_step': 10,
        'model_save_step': 100,
        'use_cuda': True,
        'model_save_path': './models_timeseries',
        'log_path': './logs_timeseries',
        'pretrained_model': None
    }
    
    def __init__(self, train_loader, test_loader=None, config=None):
        """
        Initialize TimeSeriesSolver.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader (optional)
            config: Configuration dictionary
        """
        # Update config with defaults
        if config is None:
            config = {}
        self.__dict__.update(self.DEFAULTS, **config)
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Create directories
        mkdir(self.model_save_path)
        mkdir(self.log_path)
        
        # Setup logging
        self._setup_logging()
        
        # Build model
        self.build_model()
        
        # Load pretrained model if specified
        if self.pretrained_model:
            self.load_pretrained_model()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.log_path, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def build_model(self):
        """Build the Time Series DAGMM model."""
        self.model = TimeSeriesDaGMM(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            n_gmm=self.n_gmm,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Move to GPU if available
        if self.use_cuda and torch.cuda.is_available():
            self.model.cuda()
            self.logger.info("Model moved to GPU")
        
        # Print model info
        self.print_network(self.model, 'TimeSeriesDaGMM')
    
    def print_network(self, model, name):
        """Print network architecture and parameter count."""
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"{name} Network:")
        self.logger.info(f"Number of trainable parameters: {num_params:,}")
    
    def load_pretrained_model(self):
        """Load pretrained model weights."""
        model_path = os.path.join(self.model_save_path, f'{self.pretrained_model}_timeseries_dagmm.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.logger.info(f'Loaded pretrained model: {self.pretrained_model}')
        else:
            self.logger.warning(f'Pretrained model not found: {model_path}')
    
    def to_var(self, x):
        """Convert tensor to Variable and move to GPU if available."""
        if self.use_cuda and torch.cuda.is_available():
            x = x.cuda()
        return x
    
    def train_step(self, sequences, masks):
        """
        Perform one training step.
        
        Args:
            sequences: Input sequences [batch_size, seq_len, input_dim]
            masks: Mask tensors [batch_size, seq_len]
            
        Returns:
            Dictionary with loss components
        """
        self.model.train()
        
        # Move to device
        sequences = self.to_var(sequences)
        masks = self.to_var(masks)
        
        # Forward pass
        latent, decoded, z, gamma = self.model(sequences, masks)
        
        # Compute loss
        total_loss, sample_energy, recon_error, cov_diag = self.model.loss_function(
            sequences, decoded, z, gamma, masks, self.lambda_energy, self.lambda_cov_diag
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'sample_energy': sample_energy.item(),
            'recon_error': recon_error.item(),
            'cov_diag': cov_diag.item()
        }
    
    def train(self):
        """Train the Time Series DAGMM model."""
        self.logger.info("Starting training...")
        
        # Training metrics tracking
        train_losses = []
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_losses = []
            
            # Training loop
            for batch_idx, (sequences, labels, lengths, masks) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
                
                # Training step
                losses = self.train_step(sequences, masks)
                epoch_losses.append(losses['total_loss'])
                
                # Logging
                if (batch_idx + 1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    
                    log_msg = f"Epoch [{epoch+1}/{self.num_epochs}], "
                    log_msg += f"Batch [{batch_idx+1}/{len(self.train_loader)}], "
                    log_msg += f"Time: {datetime.timedelta(seconds=int(elapsed))}, "
                    
                    for key, value in losses.items():
                        log_msg += f"{key}: {value:.4f}, "
                    
                    self.logger.info(log_msg.rstrip(', '))
            
            # Epoch statistics
            avg_epoch_loss = np.mean(epoch_losses)
            train_losses.append(avg_epoch_loss)
            
            self.logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Validation
            if self.test_loader is not None and (epoch + 1) % 5 == 0:
                val_metrics = self.validate()
                self.logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}")
            
            # Save model
            if (epoch + 1) % self.model_save_step == 0:
                self.save_model(epoch + 1)
        
        self.logger.info("Training completed!")
        
        # Plot training curves
        self.plot_training_curves(train_losses)
        
        return train_losses
    
    def validate(self):
        """Validate the model on test data."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for sequences, labels, lengths, masks in self.test_loader:
                sequences = self.to_var(sequences)
                masks = self.to_var(masks)
                
                latent, decoded, z, gamma = self.model(sequences, masks)
                
                loss, _, _, _ = self.model.loss_function(
                    sequences, decoded, z, gamma, masks, self.lambda_energy, self.lambda_cov_diag
                )
                
                batch_size = sequences.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        
        return {'loss': avg_loss}
    
    def test(self, threshold_percentile=95):
        """
        Test the model and compute anomaly detection metrics.
        
        Args:
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            Dictionary with test metrics
        """
        self.logger.info("Starting testing...")
        
        self.model.eval()
        
        # Collect statistics on training data for threshold
        train_energies = []
        train_labels = []
        
        with torch.no_grad():
            for sequences, labels, lengths, masks in tqdm(self.train_loader, desc="Computing training statistics"):
                sequences = self.to_var(sequences)
                masks = self.to_var(masks)
                
                latent, decoded, z, gamma = self.model(sequences, masks)
                
                # Compute GMM parameters
                phi, mu, cov = self.model.compute_gmm_params(z, gamma)
                
                # Compute energy for each sample
                sample_energy, _ = self.model.compute_energy(z, phi, mu, cov, size_average=False)
                
                train_energies.extend(sample_energy.cpu().numpy())
                train_labels.extend(labels.numpy())
        
        # Compute threshold
        train_energies = np.array(train_energies)
        threshold = np.percentile(train_energies, threshold_percentile)
        self.logger.info(f"Anomaly threshold (percentile {threshold_percentile}): {threshold:.4f}")
        
        # Test on test data
        test_energies = []
        test_labels = []
        test_predictions = []
        
        with torch.no_grad():
            for sequences, labels, lengths, masks in tqdm(self.test_loader, desc="Testing"):
                sequences = self.to_var(sequences)
                masks = self.to_var(masks)
                
                latent, decoded, z, gamma = self.model(sequences, masks)
                
                # Use training GMM parameters for testing
                sample_energy, _ = self.model.compute_energy(z, size_average=False)
                
                energies = sample_energy.cpu().numpy()
                test_energies.extend(energies)
                test_labels.extend(labels.numpy())
                
                # Predictions based on threshold
                predictions = (energies > threshold).astype(int)
                test_predictions.extend(predictions)
        
        # Compute metrics
        test_energies = np.array(test_energies)
        test_labels = np.array(test_labels)
        test_predictions = np.array(test_predictions)
        
        accuracy = accuracy_score(test_labels, test_predictions)
        precision, recall, f1, _ = prf(test_labels, test_predictions, average='binary')
        
        try:
            auc = roc_auc_score(test_labels, test_energies)
        except ValueError:
            auc = np.nan  # In case of single class
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'threshold': threshold
        }
        
        self.logger.info("Test Results:")
        for key, value in metrics.items():
            if not np.isnan(value):
                self.logger.info(f"{key}: {value:.4f}")
        
        # Plot results
        self.plot_test_results(test_energies, test_labels, threshold)
        
        return metrics
    
    def save_model(self, epoch):
        """Save model checkpoint."""
        model_path = os.path.join(self.model_save_path, f'epoch_{epoch}_timeseries_dagmm.pth')
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Model saved: {model_path}")
    
    def plot_training_curves(self, train_losses):
        """Plot training loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.log_path, 'training_curves.png')
        plt.savefig(plot_path)
        plt.close()
        
        self.logger.info(f"Training curves saved: {plot_path}")
    
    def plot_test_results(self, energies, labels, threshold):
        """Plot test results and energy distributions."""
        plt.figure(figsize=(15, 5))
        
        # Energy distribution
        plt.subplot(1, 3, 1)
        normal_energies = energies[labels == 0]
        anomaly_energies = energies[labels == 1]
        
        plt.hist(normal_energies, bins=50, alpha=0.7, label='Normal', density=True)
        plt.hist(anomaly_energies, bins=50, alpha=0.7, label='Anomalous', density=True)
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
        plt.xlabel('Energy')
        plt.ylabel('Density')
        plt.title('Energy Distribution')
        plt.legend()
        
        # Energy over samples
        plt.subplot(1, 3, 2)
        plt.scatter(range(len(energies)), energies, c=labels, cmap='coolwarm', alpha=0.6)
        plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
        plt.xlabel('Sample Index')
        plt.ylabel('Energy')
        plt.title('Energy vs Sample Index')
        plt.colorbar(label='Label (0=Normal, 1=Anomaly)')
        plt.legend()
        
        # ROC-like curve
        plt.subplot(1, 3, 3)
        thresholds = np.linspace(energies.min(), energies.max(), 100)
        tpr_list = []
        fpr_list = []
        
        for th in thresholds:
            predictions = (energies > th).astype(int)
            if len(np.unique(labels)) > 1:
                tn = np.sum((predictions == 0) & (labels == 0))
                fp = np.sum((predictions == 1) & (labels == 0))
                fn = np.sum((predictions == 0) & (labels == 1))
                tp = np.sum((predictions == 1) & (labels == 1))
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                tpr_list.append(tpr)
                fpr_list.append(fpr)
        
        if tpr_list and fpr_list:
            plt.plot(fpr_list, tpr_list, 'b-', label='ROC Curve')
            plt.plot([0, 1], [0, 1], 'r--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.log_path, 'test_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Test results plot saved: {plot_path}")
    
    def predict_anomaly(self, sequences, masks=None, threshold=None):
        """
        Predict anomalies for new sequences.
        
        Args:
            sequences: Input sequences [batch_size, seq_len, input_dim]
            masks: Mask tensors [batch_size, seq_len]
            threshold: Anomaly threshold (if None, uses saved threshold)
            
        Returns:
            energies: Anomaly scores
            predictions: Binary predictions
        """
        self.model.eval()
        
        with torch.no_grad():
            sequences = self.to_var(sequences)
            if masks is not None:
                masks = self.to_var(masks)
            
            latent, decoded, z, gamma = self.model(sequences, masks)
            sample_energy, _ = self.model.compute_energy(z, size_average=False)
            
            energies = sample_energy.cpu().numpy()
            
            if threshold is not None:
                predictions = (energies > threshold).astype(int)
                return energies, predictions
            else:
                return energies