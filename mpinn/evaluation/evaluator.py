"""
Performance evaluation module for MPINN.

This module handles model evaluation against the full high-fidelity dataset,
computing various performance metrics and generating comparison plots.
"""

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Union
from pathlib import Path
from collections import defaultdict
from datetime import datetime

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
        if save_dir is None:
            raise ValueError("save_dir must be provided for saving visualizations")
        self.save_dir = Path(save_dir)
        self.plots_dir = self.save_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load full HF data for evaluation
        self.full_hf_data = self.data_pipeline.prepare_data(hf_fraction=1.0)['high_fidelity']
        
        # Add tracking for progressive metrics
        self.progressive_metrics = {
            'fractions': [],
            'metrics': defaultdict(list)  # Store metrics for each feature
        }
        
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
            
        # Update progressive metrics
        self.progressive_metrics['fractions'].append(hf_fraction)
        for feature, feature_metrics in metrics.items():
            for metric_name, value in feature_metrics.items():
                self.progressive_metrics['metrics'][f"{feature}_{metric_name}"].append(value)
        
        return metrics
    
    def plot_predictions(
        self,
        hf_fraction: float,
        show: bool = False
    ):
        """
        Generate and save prediction vs actual plots.
        
        Args:
            hf_fraction: HF data fraction used in training
            show: Whether to display plots
        """
        predictions = self.model(self.full_hf_data['inputs'], training=False)
        pred_destd = self._destandardize_predictions(predictions.numpy())
        
        feature_units = {
            'energy': 'eV/atom',
            'pressure': 'GPa',
            'diffusion': 'cm²/s'
        }
        
        for feature in self.model.output_features:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Get destandardized values
            true_values = self.data_pipeline.processor._destandardize(
                self.full_hf_data['outputs'][feature],
                self.data_pipeline.processor.stats['outputs'][feature]
            )
            pred_values = pred_destd[feature]
            
            # Scatter plot
            ax.scatter(true_values, pred_values, alpha=0.5, label='Predictions')
            
            # Perfect prediction line
            min_val = min(np.min(true_values), np.min(pred_values))
            max_val = max(np.max(true_values), np.max(pred_values))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            # Metrics on original scale
            metrics = {
                'MAPE': self._calculate_mape(true_values, pred_values),
                'RMSE': self._calculate_rmse(true_values, pred_values),
                'R²': self._calculate_r2(true_values, pred_values)
            }
            
            metrics_text = '\n'.join(f'{name}: {value:.4f}' for name, value in metrics.items())
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            unit = feature_units.get(feature, '')
            ax.set_xlabel(f'True {feature.capitalize()} ({unit})')
            ax.set_ylabel(f'Predicted {feature.capitalize()} ({unit})')
            ax.set_title(f'{feature.capitalize()} Predictions (HF Fraction: {hf_fraction*100:.0f}%)')
            ax.grid(True)
            ax.legend()
            
            # Save plot with fraction in filename to avoid overwriting
            save_path = self.plots_dir / f'{feature}_predictions_hf{hf_fraction*100:.0f}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_training_progress(
        self,
        history: Dict,
        current_fraction: float,
        metrics: Dict[str, Dict[str, float]]
    ):
        """Plot progressive training metrics with accumulated data."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Loss Evolution
        if 'lf_loss' in history:
            ax1.plot(history['lf_loss'], label='LF Loss')
        
        if 'hf_losses' in history:
            for fraction, losses in history['hf_losses'].items():
                ax1.plot(losses, label=f'HF Loss ({fraction*100:.0f}%)')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        ax1.grid(True)
        ax1.legend()
        ax1.set_title('Training Loss Evolution')
        
        # 2. Alpha Evolution
        if 'alpha_history' in history:
            ax2.plot(history['alpha_history'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Alpha')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True)
        ax2.set_title('Alpha Parameter Evolution')
        
        # 3. Performance vs HF Fraction
        fractions = self.progressive_metrics['fractions']
        for feature in self.model.output_features:
            mapes = self.progressive_metrics['metrics'][f"{feature}_mape"]
            ax3.plot(fractions, mapes, 'o-', label=feature)
        
        ax3.set_xlabel('HF Data Fraction')
        ax3.set_ylabel('MAPE (%)')
        ax3.set_xscale('log')
        ax3.grid(True)
        ax3.legend()
        ax3.set_title('Performance vs HF Fraction')
        
        # Save with timestamp to keep history
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.plots_dir / f'training_progress_{timestamp}.png', dpi=300)
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
    
    def plot_timestep_comparison(self):
        """Plot HF vs LF comparison across simulation timesteps."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        features = ['energy', 'pressure', 'diffusion']
        
        for i, feature in enumerate(features):
            hf_values = self.full_hf_data['outputs'][feature]
            lf_values = self.data_pipeline.prepare_data(1.0)['low_fidelity']['outputs'][feature]
            
            axes[i].plot(hf_values, label='High Fidelity', marker='o')
            axes[i].plot(lf_values, label='Low Fidelity', marker='s')
            axes[i].set_xlabel('Simulation Step')
            axes[i].set_ylabel(feature.capitalize())
            axes[i].grid(True)
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'timestep_comparison.png', dpi=300)
        plt.close()
    
    def plot_contours(self, feature: str, phase: str = 'hf', fraction: Optional[float] = None):
        """
        Generate contour plots for predictions across input space.
        
        Args:
            feature: Feature to plot
            phase: 'lf' or 'hf'
            fraction: HF data fraction (only for HF phase)
        """
        # Create mesh grid for temperature and density using original scale
        input_stats = self.data_pipeline.processor.stats['inputs']
        temp_range = np.linspace(
            input_stats['mean'][0] - 2*input_stats['std'][0],
            input_stats['mean'][0] + 2*input_stats['std'][0],
            50
        )
        dens_range = np.linspace(
            input_stats['mean'][1] - 2*input_stats['std'][1],
            input_stats['mean'][1] + 2*input_stats['std'][1],
            50
        )
        T, D = np.meshgrid(temp_range, dens_range)
        
        # Standardize grid points for model input
        grid_points = np.column_stack((T.flatten(), D.flatten()))
        grid_points_std = (grid_points - input_stats['mean']) / input_stats['std']
        
        # Generate predictions
        predictions = self.model(grid_points_std, training=False)
        feature_idx = self.model.output_features.index(feature)
        
        # Destandardize predictions
        output_stats = self.data_pipeline.processor.stats['outputs'][feature]
        Z = predictions[:, feature_idx].numpy() * output_stats['std'] + output_stats['mean']
        Z = Z.reshape(T.shape)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(T, D, Z)
        cbar = plt.colorbar(contour)
        
        # Add units to colorbar
        feature_units = {
            'energy': 'eV/atom',
            'pressure': 'GPa',
            'diffusion': 'cm²/s'
        }
        unit = feature_units.get(feature, '')
        cbar.set_label(f'{feature.capitalize()} ({unit})')
        
        # Scatter actual data points
        if phase == 'lf':
            lf_data = self.data_pipeline.prepare_data(1.0)['low_fidelity']
            ax.scatter(
                lf_data['inputs'][:, 0] * input_stats['std'][0] + input_stats['mean'][0],
                lf_data['inputs'][:, 1] * input_stats['std'][1] + input_stats['mean'][1],
                c='red', marker='o', label='LF Data Points'
            )
            title_suffix = '(Low Fidelity)'
            save_suffix = 'lf'
        else:
            ax.scatter(
                self.full_hf_data['inputs'][:, 0] * input_stats['std'][0] + input_stats['mean'][0],
                self.full_hf_data['inputs'][:, 1] * input_stats['std'][1] + input_stats['mean'][1],
                c='red', marker='o', label='HF Data Points'
            )
            title_suffix = f'(HF Fraction: {fraction*100:.0f}%)'
            save_suffix = f'hf{fraction*100:.0f}'
        
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Density (g/cm³)')
        ax.set_title(f'{feature.capitalize()} Prediction Contours {title_suffix}')
        ax.legend()
        
        plt.tight_layout()
        save_path = self.plots_dir / f'{feature}_contours_{save_suffix}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_computational_savings(self, hf_fractions: List[float]):
        """Plot accuracy vs computational savings."""
        # Assuming computational cost is proportional to HF data fraction
        savings = [(1 - fraction) * 100 for fraction in hf_fractions]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for feature in self.model.output_features:
            mapes = []
            for fraction in hf_fractions:
                data = self.data_pipeline.prepare_data(fraction)
                predictions = self.model(data['high_fidelity']['inputs'])
                feature_idx = self.model.output_features.index(feature)
                true = data['high_fidelity']['outputs'][feature]
                pred = predictions[:, feature_idx]
                mapes.append(self._calculate_mape(true, pred))
            
            ax.plot(savings, mapes, 'o-', label=feature)
        
        ax.set_xlabel('Computational Savings (%)')
        ax.set_ylabel('MAPE (%)')
        ax.set_title('Accuracy vs Computational Savings')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'computational_savings.png', dpi=300)
        plt.close()
    
    def plot_input_space_sampling(self):
        """Plot sampling distribution in input space."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot LF points
        lf_data = self.data_pipeline.prepare_data(1.0)['low_fidelity']
        ax.scatter(
            lf_data['inputs'][:, 0],
            lf_data['inputs'][:, 1],
            c='blue',
            marker='o',
            label='LF Samples',
            alpha=0.6
        )
        
        # Plot HF points
        ax.scatter(
            self.full_hf_data['inputs'][:, 0],
            self.full_hf_data['inputs'][:, 1],
            c='red',
            marker='s',
            label='HF Samples',
            alpha=0.8
        )
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Density')
        ax.set_title('Input Space Sampling Distribution')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'input_sampling.png', dpi=300)
        plt.close()
    
    def _destandardize_predictions(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert standardized predictions back to original scale."""
        destandardized = {}
        for i, feature in enumerate(self.model.output_features):
            feature_stats = self.data_pipeline.processor.stats['outputs'][feature]
            destandardized[feature] = predictions[:, i] * feature_stats['std'] + feature_stats['mean']
        return destandardized 