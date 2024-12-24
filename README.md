# MPINN (Multi-fidelity Physics-Informed Neural Network)

A TensorFlow implementation of Multi-fidelity Physics-Informed Neural Networks for molecular dynamics simulations.

## Installation

1. Clone the repository

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package and dependencies:

```bash
pip install -e .
```

## Data Preparation

1. Create the data directories if they don't exist:

```bash
mkdir -p examples/data-one-component-system
mkdir -p examples/data-benchmark-system
```

2. Place your molecular dynamics data files in the appropriate directories:

```
examples/data-one-component-system/
├── high_fidelity/
│   ├── energy.txt
│   ├── pressure.txt
│   └── diffusion.txt
└── low_fidelity/
    ├── energy.txt
    ├── pressure.txt
    └── diffusion.txt
```

Data files should be in space-separated format with:

- First column: Temperature
- Second column: Density
- Third column: Feature value (energy/pressure/diffusion)

## Training Configuration

1. Create a results directory:

```bash
mkdir results
mkdir results/checkpoints
```

2. Modify the training script (`examples/train_mpinn_with_config.py`) to match your data:

```python
def create_config():
    return MPINNConfig(
        # Data configuration
        data_dir=Path("examples/data-one-component-system"),
        input_features=['temperature', 'density'],
        output_features=['energy', 'pressure', 'diffusion'],
        hf_fractions=[0.1, 0.2, 0.3, 0.4, 0.5, 1.0],

        # Model architecture (optional)
        hidden_layers_lf=[20, 20, 20, 20],
        hidden_layers_hf=[20, 20],
        activation='tanh',

        # Training parameters (optional)
        batch_size=32,
        lf_epochs=1000,
        hf_epochs=500,

        # Save directory
        save_dir=Path("results")
    )
```

## Training the Model

1. Run the training script:

```bash
python examples/train_mpinn_with_config.py
```

The training process consists of two phases:

1. Low-fidelity network training
2. Progressive high-fidelity training with increasing data fractions

## Results

Training results will be saved in the `results` directory:

- Model checkpoints: `results/checkpoints/`
- Visualization plots: `results/plots/`
  - Prediction vs actual plots
  - Error distributions
  - Training progress
  - Feature contour plots
  - Computational savings analysis

## Visualizations

The training process automatically generates several types of plots:

1. Training metrics (loss, alpha parameter)
2. Prediction accuracy for each feature
3. Error distributions
4. Contour plots of predictions
5. Input space sampling distribution
6. Computational savings analysis

All plots are saved in `results/plots/` with appropriate timestamps and labels. Results are overwritten when new training is run.
