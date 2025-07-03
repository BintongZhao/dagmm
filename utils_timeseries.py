import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import warnings

def setup_experiment(output_dir, experiment_name):
    """
    Setup experiment directory with timestamp.
    
    Args:
        output_dir: Base output directory
        experiment_name: Name of the experiment
        
    Returns:
        experiment_dir: Path to experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'plots'), exist_ok=True)
    
    return experiment_dir

def compute_dtw_distance(x, y):
    """
    Compute Dynamic Time Warping distance between two sequences.
    
    Args:
        x: First sequence [length1, features]
        y: Second sequence [length2, features]
        
    Returns:
        dtw_distance: DTW distance
    """
    try:
        distance, _ = fastdtw(x, y, dist=euclidean)
        return distance
    except ImportError:
        warnings.warn("fastdtw not installed, using simplified DTW-like distance")
        return compute_simplified_dtw(x, y)

def compute_simplified_dtw(x, y):
    """
    Compute simplified DTW-like distance for sequences of same length.
    
    Args:
        x: First sequence [length, features]
        y: Second sequence [length, features]
        
    Returns:
        distance: Simplified DTW distance
    """
    if len(x) != len(y):
        # Simple alignment by truncating longer sequence
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
    
    # Compute pointwise distances
    pointwise_distances = np.linalg.norm(x - y, axis=1)
    
    # Simple DTW-like computation
    dtw_matrix = np.zeros((len(x), len(y)))
    dtw_matrix[0, 0] = pointwise_distances[0]
    
    # Fill first row and column
    for i in range(1, len(x)):
        dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + pointwise_distances[i]
    
    for j in range(1, len(y)):
        dtw_matrix[0, j] = dtw_matrix[0, j-1] + pointwise_distances[j]
    
    # Fill rest of matrix
    for i in range(1, len(x)):
        for j in range(1, len(y)):
            cost = pointwise_distances[i]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    return dtw_matrix[-1, -1]

def compute_temporal_features(x, x_hat, mask=None):
    """
    Compute comprehensive temporal features for time series.
    
    Args:
        x: Original sequences [batch_size, seq_len, features]
        x_hat: Reconstructed sequences [batch_size, seq_len, features]
        mask: Mask for valid timesteps [batch_size, seq_len]
        
    Returns:
        features: Dictionary of temporal features
    """
    features = {}
    
    batch_size, seq_len, n_features = x.shape
    
    for i in range(batch_size):
        if mask is not None:
            valid_len = mask[i].sum().item()
            x_seq = x[i, :valid_len].cpu().numpy()
            x_hat_seq = x_hat[i, :valid_len].cpu().numpy()
        else:
            x_seq = x[i].cpu().numpy()
            x_hat_seq = x_hat[i].cpu().numpy()
        
        # Basic reconstruction errors
        mse = mean_squared_error(x_seq.flatten(), x_hat_seq.flatten())
        mae = mean_absolute_error(x_seq.flatten(), x_hat_seq.flatten())
        
        # DTW distance
        dtw_dist = compute_dtw_distance(x_seq, x_hat_seq)
        
        # Temporal characteristics
        # First difference (trend)
        x_diff = np.diff(x_seq, axis=0)
        x_hat_diff = np.diff(x_hat_seq, axis=0)
        trend_error = np.mean(np.abs(x_diff - x_hat_diff))
        
        # Second difference (curvature)
        x_diff2 = np.diff(x_diff, axis=0)
        x_hat_diff2 = np.diff(x_hat_diff, axis=0)
        curvature_error = np.mean(np.abs(x_diff2 - x_hat_diff2))
        
        # Frequency domain features (simplified)
        x_fft = np.fft.fft(x_seq, axis=0)
        x_hat_fft = np.fft.fft(x_hat_seq, axis=0)
        spectral_error = np.mean(np.abs(x_fft - x_hat_fft))
        
        # Correlation between channels
        if n_features > 1:
            x_corr = np.corrcoef(x_seq.T)
            x_hat_corr = np.corrcoef(x_hat_seq.T)
            corr_error = np.mean(np.abs(x_corr - x_hat_corr))
        else:
            corr_error = 0.0
        
        # Store features for this sequence
        seq_features = {
            'mse': mse,
            'mae': mae,
            'dtw_distance': dtw_dist,
            'trend_error': trend_error,
            'curvature_error': curvature_error,
            'spectral_error': spectral_error,
            'correlation_error': corr_error
        }
        
        if i == 0:
            for key in seq_features:
                features[key] = []
        
        for key, value in seq_features.items():
            features[key].append(value)
    
    # Convert to numpy arrays
    for key in features:
        features[key] = np.array(features[key])
    
    return features

def detect_anomalies_threshold(energies, threshold):
    """
    Detect anomalies using energy threshold.
    
    Args:
        energies: Anomaly scores
        threshold: Threshold value
        
    Returns:
        predictions: Binary predictions
    """
    return (energies > threshold).astype(int)

def detect_anomalies_percentile(energies, percentile=95):
    """
    Detect anomalies using percentile threshold.
    
    Args:
        energies: Anomaly scores
        percentile: Percentile for threshold
        
    Returns:
        predictions: Binary predictions
        threshold: Computed threshold
    """
    threshold = np.percentile(energies, percentile)
    predictions = detect_anomalies_threshold(energies, threshold)
    return predictions, threshold

def plot_sample_sequences(dataset, n_samples=4, save_path=None):
    """
    Plot sample sequences from dataset.
    
    Args:
        dataset: TimeSeriesDataset
        n_samples: Number of samples to plot
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Sample sequences
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        sequence, label, length = dataset[idx]
        sequence = sequence[:length]  # Remove padding
        
        # Plot first few features
        n_features_to_plot = min(5, sequence.shape[1])
        
        for j in range(n_features_to_plot):
            axes[i].plot(sequence[:, j], alpha=0.7, label=f'Feature {j+1}')
        
        axes[i].set_title(f'Sample {idx} (Label: {"Anomaly" if label == 1 else "Normal"})')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_reconstruction_comparison(original, reconstructed, mask=None, n_samples=2, save_path=None):
    """
    Plot comparison between original and reconstructed sequences.
    
    Args:
        original: Original sequences [batch_size, seq_len, features]
        reconstructed: Reconstructed sequences [batch_size, seq_len, features]
        mask: Mask for valid timesteps [batch_size, seq_len]
        n_samples: Number of samples to plot
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(n_samples, 2, figsize=(15, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        if i >= original.shape[0]:
            break
            
        if mask is not None:
            valid_len = mask[i].sum().item()
            orig_seq = original[i, :valid_len]
            recon_seq = reconstructed[i, :valid_len]
        else:
            orig_seq = original[i]
            recon_seq = reconstructed[i]
        
        # Plot first few features
        n_features_to_plot = min(5, orig_seq.shape[1])
        
        # Original
        for j in range(n_features_to_plot):
            axes[i, 0].plot(orig_seq[:, j], alpha=0.7, label=f'Feature {j+1}')
        axes[i, 0].set_title(f'Original Sequence {i+1}')
        axes[i, 0].set_xlabel('Time Steps')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Reconstructed
        for j in range(n_features_to_plot):
            axes[i, 1].plot(recon_seq[:, j], alpha=0.7, label=f'Feature {j+1}')
        axes[i, 1].set_title(f'Reconstructed Sequence {i+1}')
        axes[i, 1].set_xlabel('Time Steps')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_anomaly_scores(energies, labels, threshold=None, save_path=None):
    """
    Plot anomaly scores with labels.
    
    Args:
        energies: Anomaly scores
        labels: True labels
        threshold: Anomaly threshold
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Score distribution
    normal_scores = energies[labels == 0]
    anomaly_scores = energies[labels == 1]
    
    axes[0].hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True)
    axes[0].hist(anomaly_scores, bins=50, alpha=0.7, label='Anomalous', density=True)
    
    if threshold is not None:
        axes[0].axvline(threshold, color='red', linestyle='--', 
                       label=f'Threshold: {threshold:.4f}')
    
    axes[0].set_xlabel('Anomaly Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Anomaly Score Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scores over time
    colors = ['blue' if label == 0 else 'red' for label in labels]
    axes[1].scatter(range(len(energies)), energies, c=colors, alpha=0.6)
    
    if threshold is not None:
        axes[1].axhline(threshold, color='red', linestyle='--', 
                       label=f'Threshold: {threshold:.4f}')
    
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Anomaly Score')
    axes[1].set_title('Anomaly Scores Over Samples')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_time_series_windows(data, window_size, stride=1):
    """
    Create sliding windows from time series data.
    
    Args:
        data: Time series data [length, features]
        window_size: Size of each window
        stride: Stride between windows
        
    Returns:
        windows: List of windows
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i + window_size]
        windows.append(window)
    
    return windows

def preprocess_time_series(data, method='standard', scaler=None):
    """
    Preprocess time series data.
    
    Args:
        data: Time series data [length, features]
        method: Preprocessing method ('standard', 'minmax', 'none')
        scaler: Pre-fitted scaler (optional)
        
    Returns:
        processed_data: Preprocessed data
        scaler: Fitted scaler
    """
    if method == 'none':
        return data, None
    
    if scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
        
        scaler.fit(data)
    
    processed_data = scaler.transform(data)
    return processed_data, scaler

def evaluate_anomaly_detection(y_true, y_pred, y_scores=None):
    """
    Evaluate anomaly detection performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Anomaly scores (optional)
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['roc_auc'] = np.nan
            
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_scores)
        except ValueError:
            metrics['pr_auc'] = np.nan
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['true_positive'] = tp
    
    return metrics

def save_experiment_config(config, save_path):
    """
    Save experiment configuration to file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    import json
    
    # Convert non-serializable objects to strings
    config_serializable = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool, list, dict)):
            config_serializable[key] = value
        else:
            config_serializable[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(config_serializable, f, indent=2)

def load_experiment_config(config_path):
    """
    Load experiment configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        config: Configuration dictionary
    """
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config