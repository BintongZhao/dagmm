import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

class TimeSeriesDataset(Dataset):
    """
    Time Series Dataset for DAGMM anomaly detection.
    Supports variable length sequences with padding and masking.
    """
    
    def __init__(self, 
                 data, 
                 labels=None, 
                 sequence_length=None,
                 overlap=0.5,
                 normalization='standard',
                 mode='train'):
        """
        Initialize TimeSeriesDataset.
        
        Args:
            data: Time series data. Can be:
                  - 3D numpy array [n_sequences, max_length, n_features]
                  - List of 2D arrays [length_i, n_features]
                  - 2D numpy array [total_length, n_features] (will be windowed)
            labels: Labels for anomaly detection (0=normal, 1=anomaly)
            sequence_length: Fixed sequence length for windowing (if data is 2D)
            overlap: Overlap ratio for windowing (0 to 1)
            normalization: Type of normalization ('standard', 'minmax', or None)
            mode: 'train' or 'test'
        """
        self.mode = mode
        self.normalization = normalization
        self.sequence_length = sequence_length
        self.overlap = overlap
        
        # Process input data
        self.sequences, self.labels, self.lengths = self._process_data(data, labels)
        
        # Fit normalizer on training data
        if normalization and mode == 'train':
            self._fit_normalizer()
        
        # Apply normalization
        if normalization:
            self._apply_normalization()
    
    def _process_data(self, data, labels):
        """Process input data into sequences."""
        if isinstance(data, list):
            # List of variable length sequences
            sequences = [torch.FloatTensor(seq) for seq in data]
            lengths = [len(seq) for seq in data]
            
            if labels is not None:
                labels = torch.LongTensor(labels)
            else:
                labels = torch.zeros(len(sequences), dtype=torch.long)
                
        elif len(data.shape) == 3:
            # 3D array of sequences
            sequences = [torch.FloatTensor(data[i]) for i in range(data.shape[0])]
            lengths = [data.shape[1]] * data.shape[0]  # Assume all same length initially
            
            if labels is not None:
                labels = torch.LongTensor(labels)
            else:
                labels = torch.zeros(data.shape[0], dtype=torch.long)
                
        elif len(data.shape) == 2 and self.sequence_length is not None:
            # 2D array - create sliding windows
            sequences, labels, lengths = self._create_windows(data, labels)
            
        else:
            raise ValueError("Invalid data format. Expected 2D/3D array or list of arrays.")
        
        return sequences, labels, lengths
    
    def _create_windows(self, data, labels):
        """Create sliding windows from 2D time series data."""
        n_timesteps, n_features = data.shape
        step_size = max(1, int(self.sequence_length * (1 - self.overlap)))
        
        sequences = []
        window_labels = []
        lengths = []
        
        for start in range(0, n_timesteps - self.sequence_length + 1, step_size):
            end = start + self.sequence_length
            window = data[start:end]
            sequences.append(torch.FloatTensor(window))
            lengths.append(self.sequence_length)
            
            # Label assignment for windows
            if labels is not None:
                # If any timestep in window is anomalous, mark window as anomalous
                window_label = 1 if np.any(labels[start:end] == 1) else 0
            else:
                window_label = 0
            window_labels.append(window_label)
        
        return sequences, torch.LongTensor(window_labels), lengths
    
    def _fit_normalizer(self):
        """Fit normalizer on training sequences."""
        # Concatenate all sequences for fitting
        all_data = torch.cat(self.sequences, dim=0).numpy()
        
        if self.normalization == 'standard':
            self.scaler = StandardScaler()
        elif self.normalization == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
            return
        
        self.scaler.fit(all_data)
    
    def _apply_normalization(self):
        """Apply normalization to sequences."""
        if not hasattr(self, 'scaler') or self.scaler is None:
            return
        
        normalized_sequences = []
        for seq in self.sequences:
            seq_np = seq.numpy()
            seq_normalized = self.scaler.transform(seq_np)
            normalized_sequences.append(torch.FloatTensor(seq_normalized))
        
        self.sequences = normalized_sequences
    
    def set_scaler(self, scaler):
        """Set scaler from training dataset."""
        self.scaler = scaler
        if scaler is not None:
            self._apply_normalization()
    
    def get_scaler(self):
        """Get fitted scaler."""
        return getattr(self, 'scaler', None)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        length = self.lengths[idx]
        
        return sequence, label, length

def collate_fn(batch):
    """
    Custom collate function for variable length sequences.
    
    Args:
        batch: List of (sequence, label, length) tuples
        
    Returns:
        sequences: Padded sequences [batch_size, max_length, n_features]
        labels: Labels [batch_size]
        lengths: Sequence lengths [batch_size]
        mask: Mask for valid timesteps [batch_size, max_length]
    """
    sequences, labels, lengths = zip(*batch)
    
    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    # Create mask
    batch_size, max_length = padded_sequences.shape[:2]
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    labels = torch.stack(labels)
    lengths = torch.LongTensor(lengths)
    
    return padded_sequences, labels, lengths, mask

class SyntheticTimeSeriesGenerator:
    """Generate synthetic time series data for testing and demonstration."""
    
    @staticmethod
    def generate_normal_sequences(n_sequences=1000, 
                                  seq_length_range=(50, 200),
                                  n_features=256,
                                  noise_level=0.1):
        """
        Generate normal time series sequences.
        
        Args:
            n_sequences: Number of sequences to generate
            seq_length_range: Tuple of (min_length, max_length)
            n_features: Number of features
            noise_level: Noise level for random component
            
        Returns:
            sequences: List of sequences
            labels: List of labels (all 0 for normal)
        """
        sequences = []
        labels = []
        
        for _ in range(n_sequences):
            length = np.random.randint(seq_length_range[0], seq_length_range[1] + 1)
            
            # Generate base patterns
            t = np.linspace(0, 4 * np.pi, length)
            
            # Combine different patterns for different features
            sequence = np.zeros((length, n_features))
            
            for i in range(n_features):
                # Mix of sine waves with different frequencies and phases
                freq1 = 0.5 + np.random.random() * 2.0
                freq2 = 0.1 + np.random.random() * 0.5
                phase1 = np.random.random() * 2 * np.pi
                phase2 = np.random.random() * 2 * np.pi
                
                signal = (np.sin(freq1 * t + phase1) + 
                         0.5 * np.sin(freq2 * t + phase2) +
                         0.2 * np.sin(3 * freq1 * t + phase1))
                
                # Add trend
                trend = np.linspace(-0.5, 0.5, length) * np.random.random()
                
                # Add noise
                noise = np.random.normal(0, noise_level, length)
                
                sequence[:, i] = signal + trend + noise
            
            sequences.append(sequence)
            labels.append(0)  # Normal
        
        return sequences, labels
    
    @staticmethod
    def generate_anomalous_sequences(n_sequences=200,
                                     seq_length_range=(50, 200),
                                     n_features=256,
                                     anomaly_types=['spike', 'drift', 'pattern_change']):
        """
        Generate anomalous time series sequences.
        
        Args:
            n_sequences: Number of sequences to generate
            seq_length_range: Tuple of (min_length, max_length)
            n_features: Number of features
            anomaly_types: List of anomaly types to generate
            
        Returns:
            sequences: List of sequences
            labels: List of labels (all 1 for anomalous)
        """
        sequences = []
        labels = []
        
        for _ in range(n_sequences):
            length = np.random.randint(seq_length_range[0], seq_length_range[1] + 1)
            
            # Start with normal pattern
            normal_seq, _ = SyntheticTimeSeriesGenerator.generate_normal_sequences(
                1, (length, length), n_features, 0.1)
            sequence = normal_seq[0].copy()
            
            # Add anomaly
            anomaly_type = np.random.choice(anomaly_types)
            
            if anomaly_type == 'spike':
                # Random spikes in random features
                n_spikes = np.random.randint(1, 5)
                for _ in range(n_spikes):
                    spike_pos = np.random.randint(0, length)
                    max_spike_features = min(10, n_features)
                    n_spike_features = np.random.randint(1, max_spike_features + 1)
                    n_spike_features = min(n_spike_features, n_features)
                    spike_features = np.random.choice(n_features, n_spike_features, replace=False)
                    spike_magnitude = np.random.uniform(3, 8)
                    sequence[spike_pos, spike_features] += spike_magnitude
            
            elif anomaly_type == 'drift':
                # Gradual drift in some features
                max_drift_features = min(20, n_features)
                n_drift_features = np.random.randint(1, max_drift_features + 1)
                n_drift_features = min(n_drift_features, n_features)
                drift_features = np.random.choice(n_features, n_drift_features, replace=False)
                drift_start = np.random.randint(0, length // 2)
                drift_magnitude = np.random.uniform(1, 3)
                drift_length = length - drift_start
                
                for feat in drift_features:
                    drift = np.linspace(0, drift_magnitude, drift_length)
                    sequence[drift_start:, feat] += drift
            
            elif anomaly_type == 'pattern_change':
                # Sudden pattern change
                change_point = np.random.randint(length // 4, 3 * length // 4)
                max_change_features = min(50, n_features)
                min_change_features = min(5, n_features)
                n_change_features = np.random.randint(1, max(min_change_features + 1, max_change_features + 1))
                n_change_features = min(n_change_features, n_features)
                change_features = np.random.choice(n_features, n_change_features, replace=False)
                
                for feat in change_features:
                    # Change frequency and amplitude
                    t_change = np.linspace(0, 4 * np.pi, length - change_point)
                    new_freq = 2 + np.random.random() * 3
                    new_amplitude = 1 + np.random.random() * 2
                    new_pattern = new_amplitude * np.sin(new_freq * t_change)
                    sequence[change_point:, feat] = new_pattern
            
            sequences.append(sequence)
            labels.append(1)  # Anomalous
        
        return sequences, labels

def create_synthetic_dataset(n_normal=1000, 
                            n_anomalous=200,
                            seq_length_range=(50, 200),
                            n_features=256,
                            test_split=0.2,
                            normalization='standard'):
    """
    Create synthetic time series dataset for testing.
    
    Args:
        n_normal: Number of normal sequences
        n_anomalous: Number of anomalous sequences
        seq_length_range: Range of sequence lengths
        n_features: Number of features
        test_split: Fraction of data for testing
        normalization: Type of normalization
        
    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
    """
    # Generate sequences
    normal_seqs, normal_labels = SyntheticTimeSeriesGenerator.generate_normal_sequences(
        n_normal, seq_length_range, n_features)
    
    anomalous_seqs, anomalous_labels = SyntheticTimeSeriesGenerator.generate_anomalous_sequences(
        n_anomalous, seq_length_range, n_features)
    
    # Combine and shuffle
    all_sequences = normal_seqs + anomalous_seqs
    all_labels = normal_labels + anomalous_labels
    
    # Shuffle
    indices = np.random.permutation(len(all_sequences))
    all_sequences = [all_sequences[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Split train/test
    n_train = int(len(all_sequences) * (1 - test_split))
    
    train_sequences = all_sequences[:n_train]
    train_labels = all_labels[:n_train]
    test_sequences = all_sequences[n_train:]
    test_labels = all_labels[n_train:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_sequences, train_labels, 
                                     normalization=normalization, mode='train')
    
    test_dataset = TimeSeriesDataset(test_sequences, test_labels, 
                                    normalization=normalization, mode='test')
    
    # Share scaler from training to test
    test_dataset.set_scaler(train_dataset.get_scaler())
    
    return train_dataset, test_dataset

def get_timeseries_loader(dataset, batch_size=32, shuffle=True, num_workers=0):
    """
    Create DataLoader for time series dataset.
    
    Args:
        dataset: TimeSeriesDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader with custom collate function
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )