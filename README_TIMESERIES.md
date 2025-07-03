# Time Series DAGMM

This repository now includes a time series adaptation of the Deep Autoencoding Gaussian Mixture Model (DAGMM) for anomaly detection in sequential data.

## Original vs Time Series DAGMM

| Feature | Original DAGMM | Time Series DAGMM |
|---------|---------------|-------------------|
| Input Shape | `[batch_size, 118]` | `[batch_size, time, 256]` |
| Data Type | Static features | Time series sequences |
| Encoder | Linear layers | 2-layer LSTM |
| Decoder | Linear layers | 2-layer LSTM |
| Latent Dimension | 1D compressed + 2D errors = 3D | 8D compressed + 2D errors = 10D |
| GMM Components | 2 (configurable) | 2 (configurable) |

## Quick Start

### Basic Usage

```python
from model_timeseries import DaGMM_TimeSeries
from data_loader import get_timeseries_loader
import torch

# Create model
model = DaGMM_TimeSeries(
    input_dim=256,           # Feature dimension
    sequence_length=None,    # Variable length support
    latent_dim=8,           # Compressed representation size
    hidden_dim=64,          # LSTM hidden size
    n_gmm=2                 # Number of GMM components
)

# Create data loader with synthetic data
data_loader = get_timeseries_loader(
    batch_size=32,
    mode='train',
    sequence_length=20,      # Time steps
    input_dim=256,          # Feature dimension
    num_samples_train=1000
)

# Training step
for data, labels in data_loader:
    enc, dec, z, gamma = model(data)
    loss, energy, recon_error, cov_diag = model.loss_function(
        data, dec, z, gamma,
        lambda_energy=0.1,
        lambda_cov_diag=0.005
    )
    # Backpropagation would go here
```

### Anomaly Detection

```python
# Evaluation mode
model.eval()
with torch.no_grad():
    enc, dec, z, gamma = model(test_data)
    sample_energy, _ = model.compute_energy(z, size_average=False)
    
    # Higher energy indicates more anomalous samples
    threshold = torch.percentile(sample_energy, 80)
    anomalies = sample_energy > threshold
```

## Model Architecture

### Encoder (Time Series → Latent)
1. **Input**: `[batch_size, sequence_length, 256]`
2. **LSTM Layers**: 2-layer LSTM with hidden size 64
3. **Output Layer**: Linear layer to latent dimension (default: 8)
4. **Result**: `[batch_size, latent_dim]`

### Decoder (Latent → Time Series)
1. **Input**: `[batch_size, latent_dim]`
2. **Expansion**: Linear layer to hidden dimension
3. **LSTM Layers**: 2-layer LSTM with hidden size 64
4. **Output Layer**: Linear layer to feature dimension (256)
5. **Result**: `[batch_size, sequence_length, 256]`

### Latent Representation
- **Compressed Features**: From LSTM encoder (8D)
- **Reconstruction Errors**: 
  - Relative Euclidean distance (1D)
  - Cosine similarity (1D)
- **Total Latent**: 10D vector fed to GMM

### GMM (Gaussian Mixture Model)
- Same as original DAGMM
- Estimates density in 10D latent space
- Provides anomaly scores via sample energy

## Data Loader

The `TimeSeriesLoader` class provides:

### Synthetic Data Generation
- **Normal patterns**: Smooth sinusoidal sequences with noise
- **Anomaly patterns**: 
  - Random spikes
  - Abrupt pattern changes
- **Configurable**: Sequence length, feature dimension, sample counts

### Custom Data Support
Extend the `_load_from_file` method to load your own time series data:

```python
def _load_from_file(self, data_path):
    # Load your time series data
    # Expected format: [num_samples, sequence_length, feature_dim]
    data = np.load(data_path)  # Your loading logic
    self.data = data['sequences'].astype(np.float32)
    self.labels = data['labels'].astype(np.float32)
```

## Key Features

### ✅ Variable Length Sequences
The model can handle sequences of different lengths in the same batch by processing each sequence independently through the LSTM.

### ✅ Temporal Pattern Learning
LSTM encoder captures temporal dependencies and patterns in the sequence data.

### ✅ Reconstruction-based Anomalies
Anomalies are detected based on:
1. **Reconstruction error**: How well the model can reconstruct the input
2. **Latent density**: How likely the latent representation is under the learned GMM

### ✅ End-to-End Training
The entire model (encoder, decoder, GMM) is trained jointly with a single loss function combining reconstruction and density estimation.

## Example

Run the complete example:

```bash
python timeseries_example.py
```

This demonstrates:
- Model creation and configuration
- Training loop simulation
- Anomaly detection evaluation
- Performance metrics

## Integration with Existing Framework

The time series DAGMM maintains the same API structure as the original model, making it compatible with the existing training framework (`solver.py`). The main differences are:

1. **Import**: Use `from model_timeseries import DaGMM_TimeSeries`
2. **Data Loader**: Use `get_timeseries_loader` instead of `get_loader`
3. **Input Shape**: Ensure your data has shape `[batch_size, time, features]`

The loss function, GMM computation, and energy calculation remain identical to the original implementation.