"""
Example script demonstrating MPINN training workflow using configuration.
"""

import tensorflow as tf
from pathlib import Path
from mpinn.config import MPINNConfig
from mpinn.data import DataPipeline
from mpinn.core import MPINN
from mpinn.training import MPINNTrainer
from mpinn.visualization import TrainingMonitor

def create_config():
    """Create MPINN configuration."""
    return MPINNConfig(
        # Data configuration
        data_dir=Path("data-one-component-system"),
        input_features=['temperature', 'density'],
        output_features=['energy', 'pressure', 'diffusion'],
        hf_fractions=[0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
        
        # Model and training settings can use defaults
        save_dir=Path("results")
    )

def main():
    # Load configuration
    config = create_config()
    
    # Create output directories
    config.save_dir.mkdir(exist_ok=True)
    (config.save_dir / "checkpoints").mkdir(exist_ok=True)
    (config.save_dir / "plots").mkdir(exist_ok=True)
    
    # Initialize components using config
    data_pipeline = DataPipeline(
        data_dir=config.data_dir,
        input_features=config.input_features,
        output_features=config.output_features,
        hf_fractions=config.hf_fractions
    )
    
    model = MPINN(
        input_features=config.input_features,
        output_features=config.output_features,
        hidden_layers_lf=config.hidden_layers_lf,
        hidden_layers_hf=config.hidden_layers_hf,
        activation=config.activation,
        l2_reg=config.l2_reg
    )
    
    trainer = MPINNTrainer(
        model=model,
        data_pipeline=data_pipeline,
        optimizer_config=config.optimizer_config,
        checkpoint_dir=config.save_dir / "checkpoints"
    )
    
    monitor = TrainingMonitor(
        save_dir=config.save_dir / "plots"
    )
    
    # Training workflow
    print("Starting MPINN training...")
    
    # 1. Train low-fidelity network
    print("\nPhase 1: Low-Fidelity Training")
    trainer.train_low_fidelity(
        epochs=config.lf_epochs,
        batch_size=config.batch_size,
        patience=config.lf_patience
    )
    
    # 2. Progressive high-fidelity training
    print("\nPhase 2: Progressive High-Fidelity Training")
    trainer.train_high_fidelity(
        hf_fractions=config.hf_fractions,
        epochs_per_fraction=config.hf_epochs,
        batch_size=config.batch_size,
        patience=config.hf_patience
    )
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 