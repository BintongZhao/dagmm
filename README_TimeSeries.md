# Time Series DAGMM for Anomaly Detection

This repository extends the original DAGMM (Deep Autoencoding Gaussian Mixture Model) implementation to handle time series data with input shape `[batch_size, time, features]`. The implementation provides a complete, production-ready solution for time series anomaly detection.

## 🚀 Key Features

### Model Architecture
- **LSTM-based Encoder-Decoder**: Bidirectional LSTM encoder with unidirectional decoder for temporal modeling
- **Variable-Length Sequence Support**: Handles sequences from 10 to 1000+ timesteps with padding and masking
- **Time Series Features**: Combines latent representations with temporal reconstruction features (MSE, MAE, DTW-like, trend analysis)
- **GMM-based Anomaly Scoring**: Maintains the original DAGMM's Gaussian Mixture Model for robust anomaly detection

### Data Processing
- **Flexible Data Loading**: Support for both fixed and variable-length sequences
- **Synthetic Data Generation**: Built-in generator for testing with multiple anomaly types (spikes, drifts, pattern changes)
- **Preprocessing Pipeline**: Standard/MinMax normalization with proper train/test splitting
- **Batch Processing**: Efficient collate functions for variable-length sequences

### Training & Evaluation
- **Comprehensive Solver**: Full training pipeline with validation, checkpointing, and logging
- **Visualization Tools**: Training curves, reconstruction comparisons, and anomaly score distributions
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, AUC, and detailed error analysis
- **Experiment Management**: Organized output structure with timestamps and configuration tracking

## 📁 File Structure

```
dagmm/
├── model_timeseries.py           # Time Series DAGMM model implementation
├── data_loader_timeseries.py     # Data loading and synthetic data generation
├── solver_timeseries.py          # Training and evaluation pipeline
├── train_timeseries.py           # Command-line training script
├── utils_timeseries.py           # Utility functions and metrics
├── test_timeseries.py            # Unit tests for validation
├── TimeSeries_DAGMM_Example.ipynb # Complete demonstration notebook
└── README_TimeSeries.md          # This documentation
```

## 🔧 Installation

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install scikit-learn matplotlib pandas tqdm ipython jupyter
pip install fastdtw  # For DTW distance computation
```

### Quick Setup
```bash
git clone https://github.com/BintongZhao/dagmm.git
cd dagmm
pip install -r requirements.txt  # If available
```

## 🚦 Quick Start

### 1. Basic Training
```bash
python train_timeseries.py \
    --n_normal 1000 \
    --n_anomalous 200 \
    --seq_length_min 50 \
    --seq_length_max 200 \
    --input_dim 256 \
    --num_epochs 100 \
    --mode train_test
```

### 2. Python API Usage
```python
from model_timeseries import TimeSeriesDaGMM
from data_loader_timeseries import create_synthetic_dataset, get_timeseries_loader
from solver_timeseries import TimeSeriesSolver

# Create dataset
train_dataset, test_dataset = create_synthetic_dataset(
    n_normal=1000,
    n_anomalous=200,
    seq_length_range=(50, 200),
    n_features=256,
    normalization='standard'
)

# Create data loaders
train_loader = get_timeseries_loader(train_dataset, batch_size=32, shuffle=True)
test_loader = get_timeseries_loader(test_dataset, batch_size=32, shuffle=False)

# Configure and train model
config = {
    'input_dim': 256,
    'latent_dim': 32,
    'hidden_dim': 64,
    'num_layers': 2,
    'n_gmm': 4,
    'num_epochs': 100,
    'lr': 1e-4
}

solver = TimeSeriesSolver(train_loader, test_loader, config)
solver.train()
metrics = solver.test()
```

### 3. Custom Data Loading
```python
import numpy as np
from data_loader_timeseries import TimeSeriesDataset, get_timeseries_loader

# Load your own data
# Format: List of 2D arrays [seq_length, n_features] or 3D array [n_sequences, max_length, n_features]
your_sequences = [...]  # Your time series data
your_labels = [...]     # 0 for normal, 1 for anomalous

# Create dataset
dataset = TimeSeriesDataset(
    your_sequences, 
    your_labels, 
    normalization='standard',
    mode='train'
)

# Create loader
loader = get_timeseries_loader(dataset, batch_size=32)
```

## 🎯 Model Architecture Details

### Encoder-Decoder Structure
```
Input: [batch_size, seq_len, input_dim] → LSTM Encoder → [batch_size, latent_dim]
Latent: [batch_size, latent_dim] → LSTM Decoder → [batch_size, seq_len, input_dim]
```

### Feature Engineering
The model creates a comprehensive feature vector `z` by combining:
- **Latent features**: Compressed temporal representation from LSTM encoder
- **Reconstruction errors**: MSE, MAE between original and reconstructed sequences
- **Temporal features**: DTW-like distance, trend analysis, curvature measures

### GMM Components
- **Estimation Network**: Maps feature vector to mixture component probabilities
- **Parameter Learning**: Learns mixture weights (φ), means (μ), and covariances (Σ)
- **Energy Computation**: Computes sample energy for anomaly scoring

## 📊 Performance Metrics

The implementation provides comprehensive evaluation:
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score
- **Ranking Metrics**: AUC-ROC, AUC-PR
- **Threshold Analysis**: Percentile-based threshold selection
- **Visualization**: Energy distributions, ROC curves, confusion matrices

## ⚙️ Configuration Options

### Model Parameters
- `input_dim`: Number of features per timestep
- `latent_dim`: Dimensionality of latent representation
- `hidden_dim`: LSTM hidden state size
- `num_layers`: Number of LSTM layers
- `dropout`: Dropout rate for regularization
- `bidirectional`: Use bidirectional LSTM encoder
- `n_gmm`: Number of Gaussian mixture components

### Training Parameters
- `lr`: Learning rate
- `num_epochs`: Training epochs
- `batch_size`: Batch size
- `lambda_energy`: Weight for energy term in loss
- `lambda_cov_diag`: Weight for covariance regularization

### Data Parameters
- `seq_length_range`: (min_length, max_length) for sequences
- `normalization`: 'standard', 'minmax', or None
- `test_split`: Fraction of data for testing

## 🔬 Synthetic Data Generation

The implementation includes sophisticated synthetic data generation:

### Normal Patterns
- **Multi-frequency Sinusoids**: Realistic temporal patterns
- **Trend Components**: Linear trends with noise
- **Cross-feature Correlations**: Realistic multivariate dependencies

### Anomaly Types
- **Spikes**: Sudden amplitude increases in random features
- **Drifts**: Gradual changes in baseline values
- **Pattern Changes**: Sudden frequency/amplitude modifications

## 📈 Visualization and Analysis

### Training Monitoring
- Real-time loss tracking
- Learning curve visualization
- Parameter evolution plots

### Result Analysis
- Energy distribution comparisons
- Reconstruction quality assessment
- Feature importance analysis
- Error type breakdown

## 🧪 Testing and Validation

Run the comprehensive test suite:
```bash
python test_timeseries.py
```

This validates:
- Model creation and forward pass
- Data loading and batching
- Training pipeline
- Inference functionality

## 📝 Example Notebook

The `TimeSeries_DAGMM_Example.ipynb` provides a complete walkthrough:
1. Environment setup and data creation
2. Model architecture exploration
3. Training with monitoring
4. Comprehensive evaluation
5. Result visualization and interpretation

## 🎛️ Advanced Usage

### Custom Anomaly Types
```python
from data_loader_timeseries import SyntheticTimeSeriesGenerator

# Define custom anomaly generation
def custom_anomaly(sequence, length, n_features):
    # Your custom anomaly logic here
    return modified_sequence

# Extend the generator
SyntheticTimeSeriesGenerator.custom_anomaly = custom_anomaly
```

### Model Customization
```python
class CustomTimeSeriesDaGMM(TimeSeriesDaGMM):
    def compute_temporal_features(self, x, x_hat, mask=None):
        # Add your custom temporal features
        base_features = super().compute_temporal_features(x, x_hat, mask)
        custom_features = your_feature_computation(x, x_hat)
        return torch.cat([base_features, custom_features], dim=1)
```

## 🚨 Troubleshooting

### Common Issues
1. **NaN values during training**: Reduce learning rate or increase regularization
2. **Memory issues**: Reduce batch size or sequence length
3. **Poor convergence**: Adjust GMM components or increase model capacity

### Performance Tips
- Use GPU acceleration for large datasets
- Implement gradient clipping for stability
- Monitor covariance conditioning numbers
- Use proper data normalization

## 📚 References

1. [Original DAGMM Paper](https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf)
2. [Time Series Anomaly Detection Survey](https://arxiv.org/abs/2106.07437)
3. [LSTM for Sequence Modeling](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project extends the original DAGMM implementation. Please check the original repository for licensing terms.

## 🔗 Citation

If you use this time series extension in your research, please cite:
```bibtex
@software{timeseries_dagmm_2024,
  title={Time Series DAGMM for Anomaly Detection},
  author={Extended from original DAGMM implementation},
  year={2024},
  url={https://github.com/BintongZhao/dagmm}
}
```