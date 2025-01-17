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
        self.history = {
            'lf_loss': [],
            'hf_losses': {},  # Dictionary to store losses for each fraction
            'alpha_history': []  # Track alpha values
        }
        
        # Setup optimizers
        self.optimizer_config = optimizer_config or {
            'lf': {'optimizer': 'Adam', 'learning_rate': 0.001},
            'hf': {'optimizer': 'Adam', 'learning_rate': 0.0005}
        }
        
        self._setup_optimizers()
        
        # Initialize evaluator with plots directory
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be provided for saving results")
        
        self.evaluator = MPINNEvaluator(
            model=self.model,
            data_pipeline=self.data_pipeline,
            save_dir=self.checkpoint_dir.parent  # Use parent of checkpoint_dir
        )
        
    def _get_optimizer(self, optimizer_name: str):
        """Convert optimizer name to TensorFlow optimizer class."""
        optimizer_map = {
            'Adam': tf.keras.optimizers.Adam,
            'SGD': tf.keras.optimizers.SGD,
            'RMSprop': tf.keras.optimizers.RMSprop
        }
        if optimizer_name not in optimizer_map:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return optimizer_map[optimizer_name]
        
    def _setup_optimizers(self):
        """Configure optimizers for LF and HF training phases."""
        for phase in ['lf', 'hf']:
            config = self.optimizer_config[phase]
            optimizer_class = self._get_optimizer(config['optimizer'])
            setattr(
                self,
                f"{phase}_optimizer",
                optimizer_class(learning_rate=config['learning_rate'])
            )
    
    @tf.function
    def _train_step_lf(self, inputs, targets):
        """Single training step for low-fidelity network."""
        with tf.GradientTape() as tape:
            predictions = self.model.low_fidelity_net(inputs, training=True)
            
            # Calculate loss for each output feature
            losses = []
            for i, feature in enumerate(self.model.output_features):
                # Cast targets to float32 to match predictions
                target = tf.cast(targets[feature], tf.float32)
                feature_loss = tf.reduce_mean(tf.square(predictions[:, i] - target))
                losses.append(feature_loss)
            
            # Average loss across features
            total_loss = tf.reduce_mean(losses)
            
        gradients = tape.gradient(
            total_loss,
            self.model.low_fidelity_net.trainable_variables
        )
        self.lf_optimizer.apply_gradients(
            zip(gradients, self.model.low_fidelity_net.trainable_variables)
        )
        
        return total_loss
    
    @tf.function
    def _train_step_hf(self, inputs, targets):
        """Single training step for high-fidelity networks."""
        with tf.GradientTape(persistent=True) as tape:
            predictions = self.model(inputs, training=True)
            
            # Calculate loss for each output feature
            losses = []
            for i, feature in enumerate(self.model.output_features):
                # Cast targets to float32 to match predictions
                target = tf.cast(targets[feature], tf.float32)
                feature_loss = tf.reduce_mean(tf.square(predictions[:, i] - target))
                losses.append(feature_loss)
            
            # Average loss across features
            total_loss = tf.reduce_mean(losses)
            
            # Add L2 regularization for nonlinear network
            for layer in self.model.high_fidelity_net.nonlinear_layers:
                total_loss += tf.reduce_sum(layer.losses)
        
        # Train high-fidelity network parameters
        hf_vars = self.model.high_fidelity_net.trainable_variables
        hf_grads = tape.gradient(total_loss, hf_vars)
        self.hf_optimizer.apply_gradients(zip(hf_grads, hf_vars))
        
        # Separately update alpha
        alpha_grad = tape.gradient(total_loss, self.model.alpha)
        if alpha_grad is not None:
            new_alpha = self.model.alpha - self.hf_optimizer.learning_rate * alpha_grad
            self.model.alpha.assign(tf.clip_by_value(new_alpha, 0.0, 1.0))
        
        # Delete the tape since it's persistent
        del tape
        
        return total_loss
    
    def train_low_fidelity(
        self,
        epochs: int = 1000,
        batch_size: int = 32,
        patience: int = 50
    ):
        """Train the low-fidelity network."""
        print("\nTraining Low-Fidelity Network...")
        
        # Get LF data
        data = self.data_pipeline.prepare_data(hf_fraction=1.0)
        lf_data = data['low_fidelity']
        
        # Create dataset with proper repeat
        dataset = tf.data.Dataset.from_tensor_slices(
            (lf_data['inputs'], lf_data['outputs'])
        ).shuffle(
            buffer_size=1000,
            reshuffle_each_iteration=True
        ).batch(
            batch_size
        ).repeat()  # Add repeat() to prevent end-of-sequence errors
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        steps_per_epoch = len(lf_data['inputs']) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Train for exactly one epoch worth of steps
            for step in range(steps_per_epoch):
                inputs, targets = next(iter(dataset))
                loss = self._train_step_lf(inputs, targets)
                epoch_loss += loss
            
            epoch_loss /= steps_per_epoch
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
        
        # Generate LF contour plots
        print("\nGenerating Low-Fidelity visualizations...")
        for feature in self.model.output_features:
            self.evaluator.plot_contours(feature, phase='lf')
        
        # Freeze LF network
        self.model.freeze_low_fidelity()
    
    def train_high_fidelity(
        self,
        hf_fractions: List[float],
        epochs_per_fraction: int = 500,
        batch_size: int = 32,
        patience: int = 30
    ):
        """Progressive training of high-fidelity networks."""
        print("\nStarting Progressive High-Fidelity Training...")
        
        for fraction in sorted(hf_fractions):
            print(f"\nTraining with {fraction*100}% HF data...")
            
            # Initialize history for this fraction
            self.history['hf_losses'][fraction] = []  # Initialize empty list for this fraction
            
            # Get data for current fraction
            data = self.data_pipeline.prepare_data(hf_fraction=fraction)
            hf_data = data['high_fidelity']
            
            # Adjust batch size if needed
            n_samples = len(hf_data['inputs'])
            actual_batch_size = min(batch_size, n_samples)
            steps_per_epoch = max(1, n_samples // actual_batch_size)
            
            print(f"Number of samples: {n_samples}")
            print(f"Batch size: {actual_batch_size}")
            print(f"Steps per epoch: {steps_per_epoch}")
            
            # Create dataset with proper repeat
            dataset = tf.data.Dataset.from_tensor_slices(
                (hf_data['inputs'], hf_data['outputs'])
            ).shuffle(
                buffer_size=n_samples,
                reshuffle_each_iteration=True
            ).batch(
                actual_batch_size
            ).repeat()
            
            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs_per_fraction):
                epoch_loss = 0.0
                
                # Train for exactly one epoch worth of steps
                for step in range(steps_per_epoch):
                    inputs, targets = next(iter(dataset))
                    loss = self._train_step_hf(inputs, targets)
                    epoch_loss += loss
                
                epoch_loss /= steps_per_epoch
                self.history['hf_losses'][fraction].append(float(epoch_loss))
                self.history['alpha_history'].append(float(self.model.get_alpha()))
                
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
            
            print(f"\nGenerating visualizations for {fraction*100}% HF data...")
            
            # Performance metrics and plots for current fraction
            metrics = self.evaluator.evaluate(fraction)
            self.evaluator.plot_predictions(fraction)
            self.evaluator.plot_error_distribution(fraction)
            
            # Print metrics
            print(f"Results for {fraction*100}% HF data:")
            for feature, feature_metrics in metrics.items():
                print(f"\n{feature}:")
                for metric_name, value in feature_metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
            
            # Progressive Training Visualizations
            self.evaluator.plot_training_progress(
                history=self.history,
                current_fraction=fraction,
                metrics=metrics
            )
            
            # Generate additional visualizations
            print("\nGenerating additional visualizations...")
            self.evaluator.plot_timestep_comparison()
            
            # Generate contour plots for each feature
            for feature in self.model.output_features:
                self.evaluator.plot_contours(
                    feature=feature,
                    phase='hf',
                    fraction=fraction
                )
            
            self.evaluator.plot_computational_savings(hf_fractions)
            self.evaluator.plot_input_space_sampling()
            
            print(f"\nPlots saved in: {self.evaluator.plots_dir}")
    
    def _save_checkpoint(self, fraction: float, loss: float):
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return
            
        # Update checkpoint path to match Keras requirements
        checkpoint_path = self.checkpoint_dir / f"mpinn_hf{fraction*100:.0f}.weights.h5"
        self.model.save_weights(str(checkpoint_path))
        
        # Save training history
        history_path = self.checkpoint_dir / f"history_hf{fraction*100:.0f}.npz"
        np.savez(history_path, **self.history) 