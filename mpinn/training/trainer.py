"""
Training pipeline for MPINN.

This module implements the progressive training strategy, handling:
1. Initial low-fidelity training
2. Progressive high-fidelity training with increasing data fractions
3. Performance monitoring and checkpointing
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from collections import defaultdict
from ..evaluation.evaluator import MPINNEvaluator

class MPINNTrainer:
    """
    Manages the progressive training of MPINN models.
    
    Features:
    - Two-phase training (LF then HF)
    - Progressive HF data incorporation
    - Performance monitoring
    - Model checkpointing
    """
    
    def __init__(
        self,
        model,
        data_pipeline,
        optimizer_config: Dict = None,
        checkpoint_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: MPINN model instance
            data_pipeline: DataPipeline instance
            optimizer_config: Dictionary of optimizer configurations
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.data_pipeline = data_pipeline
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.history = defaultdict(list)
        
        # Setup optimizers
        self.optimizer_config = optimizer_config or {
            'lf': {
                'optimizer': tf.keras.optimizers.Adam,
                'learning_rate': 0.001
            },
            'hf': {
                'optimizer': tf.keras.optimizers.Adam,
                'learning_rate': 0.0005
            }
        }
        
        self._setup_optimizers()
        
        self.evaluator = MPINNEvaluator(self.model, self.data_pipeline)
        
    def _setup_optimizers(self):
        """Configure optimizers for LF and HF training phases."""
        self.lf_optimizer = self.optimizer_config['lf']['optimizer'](
            learning_rate=self.optimizer_config['lf']['learning_rate']
        )
        self.hf_optimizer = self.optimizer_config['hf']['optimizer'](
            learning_rate=self.optimizer_config['hf']['learning_rate']
        )
    
    @tf.function
    def _train_step_lf(self, inputs, targets):
        """Single training step for low-fidelity network."""
        with tf.GradientTape() as tape:
            predictions = self.model.low_fidelity_net(inputs, training=True)
            loss = tf.reduce_mean(tf.square(predictions - targets))
            
        gradients = tape.gradient(
            loss,
            self.model.low_fidelity_net.trainable_variables
        )
        self.lf_optimizer.apply_gradients(
            zip(gradients, self.model.low_fidelity_net.trainable_variables)
        )
        
        return loss
    
    @tf.function
    def _train_step_hf(self, inputs, targets):
        """Single training step for high-fidelity networks."""
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = tf.reduce_mean(tf.square(predictions - targets))
            
            # Add L2 regularization for nonlinear network
            for layer in self.model.high_fidelity_net.nonlinear_layers:
                loss += tf.reduce_sum(layer.losses)
            
        trainable_vars = (
            self.model.high_fidelity_net.trainable_variables +
            [self.model.alpha]
        )
        gradients = tape.gradient(loss, trainable_vars)
        self.hf_optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return loss
    
    def train_low_fidelity(
        self,
        epochs: int = 1000,
        batch_size: int = 32,
        patience: int = 50
    ):
        """
        Train the low-fidelity network.
        
        Args:
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            patience: Early stopping patience
        """
        print("\nTraining Low-Fidelity Network...")
        
        # Get LF data
        data = self.data_pipeline.prepare_data(hf_fraction=1.0)
        lf_data = data['low_fidelity']
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (lf_data['inputs'], lf_data['outputs'])
        ).shuffle(buffer_size=1000).batch(batch_size)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for inputs, targets in dataset:
                loss = self._train_step_lf(inputs, targets)
                epoch_loss += loss
                num_batches += 1
            
            epoch_loss /= num_batches
            self.history['lf_loss'].append(float(epoch_loss))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: LF Loss = {epoch_loss:.6f}")
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
        
        # Freeze LF network
        self.model.freeze_low_fidelity()
        print("Low-Fidelity Network Training Complete!")
    
    def train_high_fidelity(
        self,
        hf_fractions: List[float],
        epochs_per_fraction: int = 500,
        batch_size: int = 32,
        patience: int = 30
    ):
        """
        Progressive training of high-fidelity networks.
        
        Args:
            hf_fractions: List of HF data fractions to use
            epochs_per_fraction: Maximum epochs per fraction
            batch_size: Batch size for training
            patience: Early stopping patience
        """
        print("\nStarting Progressive High-Fidelity Training...")
        
        for fraction in sorted(hf_fractions):
            print(f"\nTraining with {fraction*100}% HF data...")
            
            # Get data for current fraction
            data = self.data_pipeline.prepare_data(hf_fraction=fraction)
            hf_data = data['high_fidelity']
            
            # Create dataset
            dataset = tf.data.Dataset.from_tensor_slices(
                (hf_data['inputs'], hf_data['outputs'])
            ).shuffle(buffer_size=1000).batch(batch_size)
            
            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs_per_fraction):
                epoch_loss = 0.0
                num_batches = 0
                
                for inputs, targets in dataset:
                    loss = self._train_step_hf(inputs, targets)
                    epoch_loss += loss
                    num_batches += 1
                
                epoch_loss /= num_batches
                self.history[f'hf_loss_{fraction}'].append(float(epoch_loss))
                
                if epoch % 10 == 0:
                    alpha = self.model.get_alpha()
                    print(
                        f"Epoch {epoch}: "
                        f"HF Loss = {epoch_loss:.6f}, "
                        f"Alpha = {alpha:.4f}"
                    )
                
                # Early stopping
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                    if self.checkpoint_dir:
                        self._save_checkpoint(fraction, epoch_loss)
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break
            
            print(
                f"Training complete for {fraction*100}% HF data. "
                f"Final alpha = {self.model.get_alpha():.4f}"
            )
            
            # Evaluate against full HF data
            metrics = self.evaluator.evaluate(fraction)
            self.evaluator.plot_predictions(fraction)
            self.evaluator.plot_error_distribution(fraction)
    
    def _save_checkpoint(self, fraction: float, loss: float):
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return
            
        checkpoint_path = self.checkpoint_dir / f"mpinn_hf{fraction*100:.0f}.h5"
        self.model.save_weights(str(checkpoint_path))
        
        # Save training history
        history_path = self.checkpoint_dir / f"history_hf{fraction*100:.0f}.npz"
        np.savez(history_path, **self.history) 