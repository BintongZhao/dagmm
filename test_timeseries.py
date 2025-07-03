#!/usr/bin/env python3
"""
Simple test script to validate the Time Series DAGMM implementation.
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_timeseries import TimeSeriesDaGMM
from data_loader_timeseries import create_synthetic_dataset, get_timeseries_loader
from solver_timeseries import TimeSeriesSolver

def test_model_creation():
    """Test model creation and forward pass."""
    print("Testing model creation...")
    
    model = TimeSeriesDaGMM(
        input_dim=10,
        latent_dim=8,
        hidden_dim=16,
        num_layers=1,
        n_gmm=2
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 20
    input_dim = 10
    
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    with torch.no_grad():
        latent, decoded, z, gamma = model(x, mask)
    
    assert latent.shape == (batch_size, 8)
    assert decoded.shape == (batch_size, seq_len, input_dim)
    assert z.shape[0] == batch_size
    assert gamma.shape == (batch_size, 2)
    
    print("✓ Model creation and forward pass successful")

def test_data_loader():
    """Test data loader functionality."""
    print("Testing data loader...")
    
    train_dataset, test_dataset = create_synthetic_dataset(
        n_normal=50,
        n_anomalous=10,
        seq_length_range=(20, 40),
        n_features=5,
        test_split=0.2,
        normalization='standard'
    )
    
    train_loader = get_timeseries_loader(train_dataset, batch_size=8, shuffle=True)
    
    # Test one batch
    for sequences, labels, lengths, masks in train_loader:
        assert sequences.shape[2] == 5  # n_features
        assert masks.shape == sequences.shape[:2]
        assert len(labels) == sequences.shape[0]
        break
    
    print("✓ Data loader test successful")

def test_training():
    """Test training functionality."""
    print("Testing training...")
    
    # Small dataset for quick test
    train_dataset, test_dataset = create_synthetic_dataset(
        n_normal=20,
        n_anomalous=5,
        seq_length_range=(15, 25),
        n_features=3,
        test_split=0.2,
        normalization='standard'
    )
    
    train_loader = get_timeseries_loader(train_dataset, batch_size=4, shuffle=True)
    test_loader = get_timeseries_loader(test_dataset, batch_size=4, shuffle=False)
    
    config = {
        'lr': 1e-3,
        'num_epochs': 2,  # Very short for test
        'batch_size': 4,
        'input_dim': 3,
        'latent_dim': 4,
        'hidden_dim': 8,
        'num_layers': 1,
        'dropout': 0.1,
        'bidirectional': True,
        'n_gmm': 2,
        'lambda_energy': 0.1,
        'lambda_cov_diag': 0.005,
        'log_step': 1,
        'model_save_step': 10,
        'use_cuda': False,  # Force CPU for testing
        'model_save_path': './test_models',
        'log_path': './test_logs'
    }
    
    solver = TimeSeriesSolver(train_loader, test_loader, config)
    
    # Train for just 2 epochs
    train_losses = solver.train()
    
    assert len(train_losses) == 2
    assert all(isinstance(loss, float) for loss in train_losses)
    
    print("✓ Training test successful")

def test_inference():
    """Test inference functionality."""
    print("Testing inference...")
    
    model = TimeSeriesDaGMM(
        input_dim=5,
        latent_dim=4,
        hidden_dim=8,
        num_layers=1,
        n_gmm=2
    )
    
    model.eval()
    
    # Test batch
    x = torch.randn(3, 15, 5)
    mask = torch.ones(3, 15, dtype=torch.bool)
    
    with torch.no_grad():
        latent, decoded, z, gamma = model(x, mask)
        sample_energy, _ = model.compute_energy(z, size_average=False)
        energies = sample_energy.numpy()
    
    assert len(energies) == 3
    assert all(isinstance(energy, (float, np.floating)) for energy in energies)
    
    print("✓ Inference test successful")

def main():
    """Run all tests."""
    print("Running Time Series DAGMM Tests")
    print("=" * 40)
    
    try:
        test_model_creation()
        test_data_loader()
        test_training()
        test_inference()
        
        print("\n" + "=" * 40)
        print("All tests passed! ✓")
        print("Time Series DAGMM implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)