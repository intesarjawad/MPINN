"""
Performance evaluation module for MPINN.

This module handles model evaluation against the full high-fidelity dataset,
computing various performance metrics and generating comparison plots.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
from pathlib import Path

class MPINNEvaluator:
    """
    Evaluates MPINN performance against full high-fidelity data.
    
    Features:
    - Prediction vs actual comparisons
    - Error metrics calculation
    - Performance visualization
    - Computational savings analysis
    """
    
    def __init__(
        self,
        model,
        data_pipeline,
        save_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained MPINN model
            data_pipeline: DataPipeline instance
            save_dir: Directory to save evaluation results
        """
        self.model = model
        self.data_pipeline = data_pipeline
        self.save_dir = Path(save_dir) if save_dir else None
        
        # Load full HF data for evaluation
        self.full_hf_data = self.data_pipeline.prepare_data(hf_fraction=1.0)['high_fidelity']
        
    def evaluate(
        self,
        hf_fraction: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance for a specific HF fraction.
        
        Args:
            hf_fraction: HF data fraction used in training
            
        Returns:
            Dictionary of metrics per output feature
        """
        # Get predictions
        predictions = self.model(self.full_hf_data['inputs'], training=False)
        
        # Calculate metrics for each output feature
        metrics = {}
        for i, feature in enumerate(self.model.output_features):
            true_values = self.full_hf_data['outputs'][feature]
            pred_values = predictions[:, i]
            
            metrics[feature] = {
                'mape': self._calculate_mape(true_values, pred_values),
                'rmse': self._calculate_rmse(true_values, pred_values),
                'r2': self._calculate_r2(true_values, pred_values)
            }
            
        return metrics
    
    def plot_predictions(
        self,
        hf_fraction: float,
        show: bool = True,
        save: bool = False
    ):
        """
        Generate prediction vs actual plots for all features.
        
        Args:
            hf_fraction: HF data fraction used in training
            show: Whether to display plots
            save: Whether to save plots
        """
        predictions = self.model(self.full_hf_data['inputs'], training=False)
        
        n_features = len(self.model.output_features)
        fig, axs = plt.subplots(
            1, n_features,
            figsize=(6*n_features, 5),
            squeeze=False
        )
        fig.suptitle(
            f'Predictions vs Actual (HF Fraction: {hf_fraction*100:.0f}%)',
            fontsize=14
        )
        
        for i, feature in enumerate(self.model.output_features):
            true_values = self.full_hf_data['outputs'][feature]
            pred_values = predictions[:, i]
            
            # Scatter plot
            axs[0, i].scatter(true_values, pred_values, alpha=0.5)
            
            # Perfect prediction line
            min_val = min(true_values.min(), pred_values.min())
            max_val = max(true_values.max(), pred_values.max())
            axs[0, i].plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Metrics
            metrics = {
                'MAPE': self._calculate_mape(true_values, pred_values),
                'RMSE': self._calculate_rmse(true_values, pred_values),
                'R²': self._calculate_r2(true_values, pred_values)
            }
            
            metrics_text = '\n'.join(
                f'{name}: {value:.4f}'
                for name, value in metrics.items()
            )
            axs[0, i].text(
                0.05, 0.95,
                metrics_text,
                transform=axs[0, i].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            axs[0, i].set_xlabel('True Values')
            axs[0, i].set_ylabel('Predicted Values')
            axs[0, i].set_title(feature)
            axs[0, i].grid(True)
        
        plt.tight_layout()
        
        if save and self.save_dir:
            plt.savefig(
                self.save_dir / f'predictions_hf{hf_fraction*100:.0f}.png'
            )
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    @staticmethod
    def _calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def _calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def plot_error_distribution(
        self,
        hf_fraction: float,
        show: bool = True,
        save: bool = False
    ):
        """
        Plot error distribution for each feature.
        
        Args:
            hf_fraction: HF data fraction used in training
            show: Whether to display plots
            save: Whether to save plots
        """
        predictions = self.model(self.full_hf_data['inputs'], training=False)
        
        n_features = len(self.model.output_features)
        fig, axs = plt.subplots(
            1, n_features,
            figsize=(6*n_features, 5),
            squeeze=False
        )
        fig.suptitle(
            f'Error Distribution (HF Fraction: {hf_fraction*100:.0f}%)',
            fontsize=14
        )
        
        for i, feature in enumerate(self.model.output_features):
            true_values = self.full_hf_data['outputs'][feature]
            pred_values = predictions[:, i]
            
            # Calculate percentage errors
            errors = ((pred_values - true_values) / true_values) * 100
            
            # Plot histogram
            axs[0, i].hist(errors, bins=50, density=True)
            axs[0, i].set_xlabel('Percentage Error (%)')
            axs[0, i].set_ylabel('Density')
            axs[0, i].set_title(feature)
            axs[0, i].grid(True)
            
            # Add statistics
            stats = {
                'Mean': np.mean(errors),
                'Std': np.std(errors),
                'Median': np.median(errors)
            }
            
            stats_text = '\n'.join(
                f'{name}: {value:.2f}%'
                for name, value in stats.items()
            )
            axs[0, i].text(
                0.95, 0.95,
                stats_text,
                transform=axs[0, i].transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        plt.tight_layout()
        
        if save and self.save_dir:
            plt.savefig(
                self.save_dir / f'errors_hf{hf_fraction*100:.0f}.png'
            )
        
        if show:
            plt.show()
        else:
            plt.close() 