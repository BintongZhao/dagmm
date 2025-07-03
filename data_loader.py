import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
import pandas as pd

class KDD99Loader(object):
    def __init__(self, data_path, mode="train"):
        self.mode=mode
        data = np.load(data_path)

        labels = data["kdd"][:,-1]
        features = data["kdd"][:,:-1]
        N, D = features.shape
        
        normal_data = features[labels==1]
        normal_labels = labels[labels==1]

        N_normal = normal_data.shape[0]

        attack_data = features[labels==0]
        attack_labels = labels[labels==0]

        N_attack = attack_data.shape[0]

        randIdx = np.arange(N_attack)
        np.random.shuffle(randIdx)
        N_train = N_attack // 2

        self.train = attack_data[randIdx[:N_train]]
        self.train_labels = attack_labels[randIdx[:N_train]]

        self.test = attack_data[randIdx[N_train:]]
        self.test_labels = attack_labels[randIdx[N_train:]]

        self.test = np.concatenate((self.test, normal_data),axis=0)
        self.test_labels = np.concatenate((self.test_labels, normal_labels),axis=0)


    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]


    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
           return np.float32(self.test[index]), np.float32(self.test_labels[index])
        

class TimeSeriesLoader(object):
    """
    Data loader for time series data with shape [batch_size, time, features]
    Generates synthetic time series data for testing or can load from file
    """
    def __init__(self, data_path=None, mode="train", sequence_length=20, input_dim=256, 
                 num_samples_train=1000, num_samples_test=200, anomaly_ratio=0.1):
        self.mode = mode
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.num_samples_train = num_samples_train
        self.num_samples_test = num_samples_test
        self.anomaly_ratio = anomaly_ratio
        
        if data_path and os.path.exists(data_path):
            # Load from file if available
            self._load_from_file(data_path)
        else:
            # Generate synthetic time series data
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic time series data for testing"""
        np.random.seed(42)  # For reproducibility
        
        if self.mode == "train":
            num_samples = self.num_samples_train
            # Generate mostly normal data for training
            num_normal = int(num_samples * (1 - self.anomaly_ratio))
            num_anomaly = num_samples - num_normal
        else:
            num_samples = self.num_samples_test
            # Generate mixed data for testing
            num_normal = int(num_samples * 0.7)  # 70% normal in test
            num_anomaly = num_samples - num_normal
        
        # Generate normal time series (smooth patterns)
        normal_data = []
        normal_labels = []
        
        for i in range(num_normal):
            # Create smooth sinusoidal patterns with some noise
            t = np.linspace(0, 4*np.pi, self.sequence_length)
            frequencies = np.random.uniform(0.5, 2.0, self.input_dim)
            phases = np.random.uniform(0, 2*np.pi, self.input_dim)
            amplitudes = np.random.uniform(0.5, 1.5, self.input_dim)
            
            # Create multi-dimensional sinusoidal time series
            series = np.zeros((self.sequence_length, self.input_dim))
            for dim in range(self.input_dim):
                series[:, dim] = amplitudes[dim] * np.sin(frequencies[dim] * t + phases[dim])
            
            # Add small amount of gaussian noise
            series += np.random.normal(0, 0.1, series.shape)
            
            normal_data.append(series)
            normal_labels.append(1)  # 1 = normal
        
        # Generate anomalous time series (irregular patterns)
        anomaly_data = []
        anomaly_labels = []
        
        for i in range(num_anomaly):
            # Create anomalous patterns
            if np.random.random() < 0.5:
                # Type 1: Random spikes
                series = np.random.normal(0, 0.2, (self.sequence_length, self.input_dim))
                # Add random spikes
                spike_positions = np.random.choice(self.sequence_length, 
                                                 size=max(1, self.sequence_length // 10), 
                                                 replace=False)
                for pos in spike_positions:
                    series[pos] += np.random.normal(0, 2, self.input_dim)
            else:
                # Type 2: Abrupt changes
                series = np.zeros((self.sequence_length, self.input_dim))
                change_point = np.random.randint(self.sequence_length // 4, 3 * self.sequence_length // 4)
                
                # Before change point: normal pattern
                t1 = np.linspace(0, 2*np.pi, change_point)
                for dim in range(self.input_dim):
                    series[:change_point, dim] = np.sin(2 * t1 + dim * 0.1)
                
                # After change point: different pattern
                t2 = np.linspace(0, 2*np.pi, self.sequence_length - change_point)
                for dim in range(self.input_dim):
                    series[change_point:, dim] = 3 * np.sin(0.5 * t2 + dim * 0.2)
                
                # Add noise
                series += np.random.normal(0, 0.3, series.shape)
            
            anomaly_data.append(series)
            anomaly_labels.append(0)  # 0 = anomaly
        
        # Combine data
        all_data = normal_data + anomaly_data
        all_labels = normal_labels + anomaly_labels
        
        # Convert to numpy arrays
        self.data = np.array(all_data, dtype=np.float32)
        self.labels = np.array(all_labels, dtype=np.float32)
        
        # Shuffle the data
        indices = np.random.permutation(len(self.data))
        self.data = self.data[indices]
        self.labels = self.labels[indices]
        
        print(f"Generated {len(self.data)} {self.mode} samples:")
        print(f"  - Normal samples: {np.sum(self.labels == 1)}")
        print(f"  - Anomaly samples: {np.sum(self.labels == 0)}")
        print(f"  - Data shape: {self.data.shape}")
    
    def _load_from_file(self, data_path):
        """Load time series data from file"""
        # This can be implemented to load actual time series data
        # For now, fall back to synthetic data generation
        print(f"File {data_path} loading not implemented, generating synthetic data instead")
        self._generate_synthetic_data()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def get_loader(data_path, batch_size, mode='train'):
    """Build and return data loader."""

    dataset = KDD99Loader(data_path, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader


def get_timeseries_loader(data_path=None, batch_size=32, mode='train', 
                         sequence_length=20, input_dim=256,
                         num_samples_train=1000, num_samples_test=200):
    """Build and return time series data loader."""
    
    dataset = TimeSeriesLoader(
        data_path=data_path,
        mode=mode,
        sequence_length=sequence_length,
        input_dim=input_dim,
        num_samples_train=num_samples_train,
        num_samples_test=num_samples_test
    )
    
    shuffle = (mode == 'train')
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    return data_loader
