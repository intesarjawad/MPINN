"""
Core MPINN model implementation.

This module implements the Multi-fidelity Physics-Informed Neural Network (MPINN)
that combines low-fidelity and high-fidelity data for molecular dynamics predictions.
"""

import tensorflow as tf
from typing import List, Dict, Optional, Union

from .networks import LowFidelityNetwork, HighFidelityNetwork

class MPINN(tf.keras.Model):
    """
    Multi-fidelity Physics-Informed Neural Network.
    
    This model combines:
    1. A low-fidelity network for base predictions
    2. High-fidelity correction networks (linear + nonlinear)
    3. A trainable parameter α for weighting corrections
    
    The model progressively learns from increasing amounts of HF data
    while maintaining the knowledge learned from LF data.
    """
    
    def __init__(
        self,
        input_features: List[str],
        output_features: List[str],
        hidden_layers_lf: List[int] = [20, 20, 20, 20],
        hidden_layers_hf: List[int] = [20, 20],
        activation: str = 'tanh',
        l2_reg: float = 0.001
    ):
        """
        Initialize the MPINN model.
        
        Args:
            input_features: List of input feature names
            output_features: List of output feature names
            hidden_layers_lf: Hidden layer sizes for LF network
            hidden_layers_hf: Hidden layer sizes for HF network
            activation: Activation function to use
            l2_reg: L2 regularization factor for nonlinear network
        """
        super().__init__()
        
        self.input_features = input_features
        self.output_features = output_features
        
        # Network dimensions
        self.input_dim = len(input_features)
        self.output_dim = len(output_features)
        
        # Initialize networks
        self.low_fidelity_net = LowFidelityNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=hidden_layers_lf,
            activation=activation
        )
        
        # HF network takes both inputs and LF predictions
        self.high_fidelity_net = HighFidelityNetwork(
            input_dim=self.input_dim + self.output_dim,
            output_dim=self.output_dim,
            hidden_layers=hidden_layers_hf,
            activation=activation,
            l2_reg=l2_reg
        )
        
        # Initialize alpha parameter [0,1]
        self.alpha = tf.Variable(
            0.5,
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, 0, 1),
            name='alpha'
        )
        
    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass through the MPINN model.
        
        Args:
            inputs: Input features tensor
            training: Whether in training mode
            
        Returns:
            Final predictions combining LF and HF networks
        """
        # Get low-fidelity predictions
        lf_predictions = self.low_fidelity_net(inputs, training=training)
        
        # Combine inputs with LF predictions for HF network
        hf_inputs = tf.concat([inputs, lf_predictions], axis=1)
        
        # Get high-fidelity corrections using constrained alpha
        hf_corrections = self.high_fidelity_net(
            hf_inputs,
            self.alpha_value,  # Use the constrained property
            training=training
        )
        
        return lf_predictions + hf_corrections
    
    def freeze_low_fidelity(self):
        """Freeze the low-fidelity network after initial training."""
        self.low_fidelity_net.trainable = False 

    def get_alpha(self) -> float:
        """
        Get current value of the alpha parameter.
        
        Returns:
            Current alpha value as a float between 0 and 1
        """
        return float(self.alpha.numpy())

    @property
    def alpha_value(self) -> tf.Tensor:
        """
        Property that ensures alpha is always properly constrained.
        
        Returns:
            Tensor containing current alpha value, guaranteed to be between 0 and 1
        """
        # Apply sigmoid to ensure soft constraints
        # Then clip to ensure hard constraints
        return tf.clip_by_value(
            tf.sigmoid(self.alpha),
            0.0,
            1.0
        ) 