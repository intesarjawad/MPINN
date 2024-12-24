"""
Example script demonstrating MPINN training workflow.
"""

import tensorflow as tf
from pathlib import Path
from mpinn.data import DataPipeline
from mpinn.core import MPINN
from mpinn.training import MPINNTrainer
from mpinn.visualization import TrainingMonitor

def main():
    # Configuration
    data_dir = Path("data")
    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True)
    
    input_features = ['temperature', 'density']
    output_features = ['energy', 'pressure', 'diffusion']
    hf_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    
    # Initialize components
    data_pipeline = DataPipeline(
        data_dir=data_dir,
        input_features=input_features,
        output_features=output_features,
        hf_fractions=hf_fractions
    )
    
    model = MPINN(
        input_features=input_features,
        output_features=output_features,
        hidden_layers_lf=[20, 20, 20, 20],
        hidden_layers_hf=[20, 20],
        activation='tanh',
        l2_reg=0.001
    )
    
    trainer = MPINNTrainer(
        model=model,
        data_pipeline=data_pipeline,
        checkpoint_dir=save_dir / "checkpoints"
    )
    
    monitor = TrainingMonitor(
        save_dir=save_dir / "plots"
    )
    
    # Training workflow
    print("Starting MPINN training...")
    
    # 1. Train low-fidelity network
    print("\nPhase 1: Low-Fidelity Training")
    trainer.train_low_fidelity(
        epochs=1000,
        batch_size=32,
        patience=50
    )
    
    # 2. Progressive high-fidelity training
    print("\nPhase 2: Progressive High-Fidelity Training")
    trainer.train_high_fidelity(
        hf_fractions=hf_fractions,
        epochs_per_fraction=500,
        batch_size=32,
        patience=30
    )
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 