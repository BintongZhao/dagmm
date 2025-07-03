#!/usr/bin/env python3
"""
Time Series DAGMM Training Script

This script demonstrates how to train and evaluate the Time Series DAGMM model
for anomaly detection on time series data.
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from data_loader_timeseries import (
    create_synthetic_dataset, 
    get_timeseries_loader,
    TimeSeriesDataset
)
from solver_timeseries import TimeSeriesSolver
from utils_timeseries import setup_experiment, plot_sample_sequences
import logging

def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Time Series DAGMM for Anomaly Detection')
    
    # Data parameters
    parser.add_argument('--data_type', type=str, default='synthetic', 
                       choices=['synthetic', 'custom'],
                       help='Type of dataset to use')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to custom dataset (for custom data type)')
    parser.add_argument('--n_normal', type=int, default=1000,
                       help='Number of normal sequences (synthetic data)')
    parser.add_argument('--n_anomalous', type=int, default=200,
                       help='Number of anomalous sequences (synthetic data)')
    parser.add_argument('--seq_length_min', type=int, default=50,
                       help='Minimum sequence length')
    parser.add_argument('--seq_length_max', type=int, default=200,
                       help='Maximum sequence length')
    parser.add_argument('--input_dim', type=int, default=256,
                       help='Number of features/input dimension')
    parser.add_argument('--test_split', type=float, default=0.2,
                       help='Fraction of data for testing')
    parser.add_argument('--normalization', type=str, default='standard',
                       choices=['standard', 'minmax', 'none'],
                       help='Data normalization method')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--bidirectional', type=str2bool, default=True,
                       help='Use bidirectional LSTM')
    parser.add_argument('--n_gmm', type=int, default=4,
                       help='Number of GMM components')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--lambda_energy', type=float, default=0.1,
                       help='Weight for energy term in loss')
    parser.add_argument('--lambda_cov_diag', type=float, default=0.005,
                       help='Weight for covariance diagonal regularization')
    
    # Logging and saving
    parser.add_argument('--log_step', type=int, default=10,
                       help='Logging frequency (batches)')
    parser.add_argument('--model_save_step', type=int, default=25,
                       help='Model saving frequency (epochs)')
    parser.add_argument('--experiment_name', type=str, default='timeseries_dagmm',
                       help='Experiment name for logging')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                       help='Output directory for experiments')
    
    # Runtime parameters
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'train_test'],
                       help='Mode: train only, test only, or both')
    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='Path to pretrained model (for testing or fine-tuning)')
    parser.add_argument('--use_cuda', type=str2bool, default=True,
                       help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loader workers')
    
    # Anomaly detection parameters
    parser.add_argument('--threshold_percentile', type=float, default=95,
                       help='Percentile for anomaly threshold')
    
    return parser.parse_args()

def load_custom_data(data_path, test_split=0.2, normalization='standard'):
    """
    Load custom time series data.
    
    Expected format: 
    - NumPy file (.npz) with keys 'sequences' and 'labels'
    - sequences: List of 2D arrays or 3D array [n_sequences, max_length, n_features]  
    - labels: 1D array [n_sequences] with 0=normal, 1=anomalous
    
    Args:
        data_path: Path to data file
        test_split: Fraction for test split
        normalization: Normalization method
        
    Returns:
        train_dataset, test_dataset
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    data = np.load(data_path, allow_pickle=True)
    sequences = data['sequences']
    labels = data['labels']
    
    # Convert to list if needed
    if isinstance(sequences, np.ndarray) and len(sequences.shape) == 3:
        sequences = [sequences[i] for i in range(sequences.shape[0])]
    
    # Split data
    n_total = len(sequences)
    n_train = int(n_total * (1 - test_split))
    
    # Shuffle indices
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_sequences = [sequences[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_sequences = [sequences[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_sequences, train_labels, 
                                     normalization=normalization, mode='train')
    test_dataset = TimeSeriesDataset(test_sequences, test_labels, 
                                    normalization=normalization, mode='test')
    
    # Share scaler
    test_dataset.set_scaler(train_dataset.get_scaler())
    
    return train_dataset, test_dataset

def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup experiment directory
    experiment_dir = setup_experiment(args.output_dir, args.experiment_name)
    
    # Setup logging
    log_file = os.path.join(experiment_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("Time Series DAGMM Training")
    logger.info("=" * 50)
    
    # Log configuration
    logger.info("Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 50)
    
    # Load or create dataset
    if args.data_type == 'synthetic':
        logger.info("Creating synthetic dataset...")
        normalization = None if args.normalization == 'none' else args.normalization
        
        train_dataset, test_dataset = create_synthetic_dataset(
            n_normal=args.n_normal,
            n_anomalous=args.n_anomalous,
            seq_length_range=(args.seq_length_min, args.seq_length_max),
            n_features=args.input_dim,
            test_split=args.test_split,
            normalization=normalization
        )
        
        logger.info(f"Created synthetic dataset:")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Test samples: {len(test_dataset)}")
        
    elif args.data_type == 'custom':
        if args.data_path is None:
            raise ValueError("data_path must be provided for custom data type")
        
        logger.info(f"Loading custom dataset from {args.data_path}...")
        normalization = None if args.normalization == 'none' else args.normalization
        
        train_dataset, test_dataset = load_custom_data(
            args.data_path, 
            args.test_split, 
            normalization
        )
        
        logger.info(f"Loaded custom dataset:")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = get_timeseries_loader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    test_loader = get_timeseries_loader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Plot sample sequences for visualization
    plot_sample_sequences(train_dataset, save_path=os.path.join(experiment_dir, 'sample_sequences.png'))
    
    # Configure solver
    config = {
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'input_dim': args.input_dim,
        'latent_dim': args.latent_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'bidirectional': args.bidirectional,
        'n_gmm': args.n_gmm,
        'lambda_energy': args.lambda_energy,
        'lambda_cov_diag': args.lambda_cov_diag,
        'log_step': args.log_step,
        'model_save_step': args.model_save_step,
        'use_cuda': args.use_cuda,
        'model_save_path': os.path.join(experiment_dir, 'models'),
        'log_path': experiment_dir,
        'pretrained_model': args.pretrained_model
    }
    
    # Create solver
    solver = TimeSeriesSolver(train_loader, test_loader, config)
    
    # Training
    if args.mode in ['train', 'train_test']:
        logger.info("Starting training...")
        train_losses = solver.train()
        logger.info("Training completed!")
        
        # Save final model
        solver.save_model('final')
    
    # Testing
    if args.mode in ['test', 'train_test']:
        logger.info("Starting testing...")
        test_metrics = solver.test(threshold_percentile=args.threshold_percentile)
        logger.info("Testing completed!")
        
        # Save test results
        results_file = os.path.join(experiment_dir, 'test_results.txt')
        with open(results_file, 'w') as f:
            f.write("Test Results:\n")
            f.write("=" * 30 + "\n")
            for key, value in test_metrics.items():
                if not np.isnan(value):
                    f.write(f"{key}: {value:.4f}\n")
        
        logger.info(f"Test results saved to: {results_file}")
    
    logger.info("Experiment completed successfully!")
    logger.info(f"Results saved in: {experiment_dir}")

if __name__ == '__main__':
    main()