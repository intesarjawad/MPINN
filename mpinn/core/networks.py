"""
Neural network architectures for MPINN.

This module defines the core network components:
1. Low-fidelity network for base predictions
2. High-fidelity correction networks (linear + nonlinear)
"""

import tensorflow as tf
from typing import List, Dict, Optional, Union

class LowFidelityNetwork(tf.keras.Model):
    """
    Base network for learning from low-fidelity data.
    
    This network learns the initial mapping from input features to outputs
    using only low-fidelity data. It will be frozen after initial training
    to serve as a foundation for high-fidelity corrections.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int] = [20, 20, 20, 20],
        activation: str = 'tanh'
    ):
        """
        Initialize the low-fidelity network.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            hidden_layers: List of hidden layer sizes
            activation: Activation function to use
        """
        super().__init__()
        
        self.layers_list = []
        
        # Input layer
        current_dim = input_dim
        
        # Hidden layers
        for units in hidden_layers:
            self.layers_list.append(
                tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    kernel_initializer='glorot_normal'
                )
            )
            current_dim = units
        
        # Output layer
        self.layers_list.append(
            tf.keras.layers.Dense(
                output_dim,
                activation='linear',
                kernel_initializer='glorot_normal'
            )
        )
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass through the network."""
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x


class HighFidelityNetwork(tf.keras.Model):
    """
    Correction networks for high-fidelity adjustments.
    
    This network consists of two components:
    1. Linear correction network
    2. Nonlinear correction network
    
    These are combined using a trainable parameter α to produce
    the final high-fidelity corrections.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int] = [20, 20],
        activation: str = 'tanh',
        l2_reg: float = 0.001
    ):
        """
        Initialize the high-fidelity correction networks.
        
        Args:
            input_dim: Number of input features (includes LF predictions)
            output_dim: Number of output features
            hidden_layers: List of hidden layer sizes for nonlinear network
            activation: Activation function for nonlinear network
            l2_reg: L2 regularization factor for nonlinear network
        """
        super().__init__()
        
        # Linear correction network (simple linear layer)
        self.linear_net = tf.keras.layers.Dense(
            output_dim,
            activation='linear',
            kernel_initializer='glorot_normal',
            name='linear_correction'
        )
        
        # Nonlinear correction network
        self.nonlinear_layers = []
        
        current_dim = input_dim
        for units in hidden_layers:
            self.nonlinear_layers.append(
                tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
                )
            )
            current_dim = units
            
        # Output layer for nonlinear network
        self.nonlinear_layers.append(
            tf.keras.layers.Dense(
                output_dim,
                activation='linear',
                kernel_initializer='glorot_normal',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name='nonlinear_correction'
            )
        )
        
    def call(
        self,
        inputs: tf.Tensor,
        alpha: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass through both correction networks.
        
        Args:
            inputs: Combined input features and LF predictions
            alpha: Trainable weight parameter [0,1]
            training: Whether in training mode
            
        Returns:
            Weighted combination of linear and nonlinear corrections
        """
        # Linear correction
        linear_correction = self.linear_net(inputs)
        
        # Nonlinear correction
        x = inputs
        for layer in self.nonlinear_layers:
            x = layer(x)
        nonlinear_correction = x
        
        # Ensure alpha is properly bounded
        alpha = tf.clip_by_value(alpha, 0.0, 1.0)
        
        # Combine corrections using alpha (sum of weights = 1)
        return alpha * linear_correction + (1 - alpha) * nonlinear_correction 