#!/usr/bin/env python3
"""
Time Series DAGMM Example
==========================

This script demonstrates how to use the Time Series DAGMM model for anomaly detection
in time series data with shape [batch_size, time, 256].

Example usage:
    python timeseries_example.py
"""

import torch
import numpy as np
from model_timeseries import DaGMM_TimeSeries
from data_loader import get_timeseries_loader

def main():
    print("Time Series DAGMM Anomaly Detection Example")
    print("=" * 50)
    
    # Model parameters
    INPUT_DIM = 256
    SEQUENCE_LENGTH = 20
    LATENT_DIM = 8
    HIDDEN_DIM = 64
    N_GMM = 2
    BATCH_SIZE = 16
    
    # Training parameters
    LAMBDA_ENERGY = 0.1
    LAMBDA_COV_DIAG = 0.005
    
    print(f"Model configuration:")
    print(f"  - Input dimension: {INPUT_DIM}")
    print(f"  - Sequence length: {SEQUENCE_LENGTH}")
    print(f"  - Latent dimension: {LATENT_DIM}")
    print(f"  - Hidden dimension: {HIDDEN_DIM}")
    print(f"  - Number of GMM components: {N_GMM}")
    print()
    
    # Create model
    model = DaGMM_TimeSeries(
        input_dim=INPUT_DIM,
        sequence_length=None,  # Variable length
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_gmm=N_GMM
    )
    
    print("✓ Time Series DAGMM model created")
    
    # Create data loaders
    train_loader = get_timeseries_loader(
        batch_size=BATCH_SIZE,
        mode='train',
        sequence_length=SEQUENCE_LENGTH,
        input_dim=INPUT_DIM,
        num_samples_train=200
    )
    
    test_loader = get_timeseries_loader(
        batch_size=BATCH_SIZE,
        mode='test',
        sequence_length=SEQUENCE_LENGTH,
        input_dim=INPUT_DIM,
        num_samples_test=100
    )
    
    print("✓ Data loaders created")
    print()
    
    # Training simulation (few steps)
    print("Training simulation (5 batches):")
    model.train()
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Forward pass
        enc, dec, z, gamma = model(data)
        
        # Compute loss
        loss, sample_energy, recon_error, cov_diag = model.loss_function(
            data, dec, z, gamma, LAMBDA_ENERGY, LAMBDA_COV_DIAG
        )
        
        print(f"  Batch {batch_idx + 1:2d}: "
              f"Loss={loss.item():8.4f}, "
              f"Recon={recon_error.item():.4f}, "
              f"Energy={sample_energy.item():6.4f}")
        
        if batch_idx >= 4:  # Train for 5 batches
            break
    
    print("✓ Training simulation completed")
    print()
    
    # Evaluation
    print("Evaluation (anomaly detection):")
    model.eval()
    
    all_energies = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            enc, dec, z, gamma = model(data)
            
            # Compute sample energies for anomaly detection
            sample_energy, _ = model.compute_energy(z, size_average=False)
            
            all_energies.extend(sample_energy.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_energies = np.array(all_energies)
    all_labels = np.array(all_labels)
    
    # Compute statistics
    normal_energies = all_energies[all_labels == 1]
    anomaly_energies = all_energies[all_labels == 0]
    
    print(f"  Normal samples ({len(normal_energies)}): "
          f"Energy = {normal_energies.mean():.4f} ± {normal_energies.std():.4f}")
    print(f"  Anomaly samples ({len(anomaly_energies)}): "
          f"Energy = {anomaly_energies.mean():.4f} ± {anomaly_energies.std():.4f}")
    
    # Simple threshold-based detection
    threshold = np.percentile(all_energies, 80)  # 80th percentile as threshold
    predictions = (all_energies > threshold).astype(int)
    
    # Calculate accuracy (considering 0=anomaly, 1=normal)
    # So high energy should predict anomaly (0)
    predictions = 1 - predictions  # Flip: high energy -> anomaly (0)
    
    accuracy = np.mean(predictions == all_labels)
    
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    print()
    print("✅ Time Series DAGMM anomaly detection completed!")
    print()
    print("Key features demonstrated:")
    print("  ✓ Handles time series input [batch_size, time, 256]")
    print("  ✓ Variable sequence length support")
    print("  ✓ LSTM-based temporal encoding/decoding")
    print("  ✓ GMM-based anomaly detection")
    print("  ✓ Reconstruction error computation")
    print("  ✓ Energy-based anomaly scoring")

if __name__ == "__main__":
    main()