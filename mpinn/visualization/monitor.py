"""
Real-time monitoring and visualization for MPINN training.

This module provides tools for:
1. Training progress monitoring
2. Performance visualization
3. Alpha parameter tracking
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
from pathlib import Path

class TrainingMonitor:
    """
    Real-time monitoring of MPINN training progress.
    
    Features:
    - Loss tracking
    - Alpha evolution
    - Performance metrics
    - Progress visualization
    """
    
    def __init__(
        self,
        save_dir: Optional[Union[str, Path]] = None,
        figure_size: tuple = (12, 8)
    ):
        """
        Initialize the monitor.
        
        Args:
            save_dir: Directory to save plots
            figure_size: Default figure size for plots
        """
        self.save_dir = Path(save_dir) if save_dir else None
        self.figure_size = figure_size
        self.history = {
            'lf_loss': [],
            'hf_losses': {},
            'alpha_history': [],
            'metrics': defaultdict(list)
        }
    
    def update_progress(
        self,
        epoch: int,
        losses: Dict[str, float],
        alpha: float,
        hf_fraction: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Update training progress and visualizations.
        
        Args:
            epoch: Current epoch number
            losses: Dictionary of current losses
            alpha: Current alpha value
            hf_fraction: Current HF data fraction
            metrics: Optional performance metrics
        """
        # Update history
        if hf_fraction is None:
            self.history['lf_loss'].append(sum(losses.values()))
        else:
            if hf_fraction not in self.history['hf_losses']:
                self.history['hf_losses'][hf_fraction] = []
            self.history['hf_losses'][hf_fraction].append(sum(losses.values()))
        
        self.history['alpha_history'].append(alpha)
        
        if metrics:
            for name, value in metrics.items():
                self.history['metrics'][name].append(value)
        
        # Display progress
        self._display_progress(epoch, losses, alpha, hf_fraction)
        
        # Update plots every N epochs
        if epoch % 10 == 0:
            self.plot_training_progress(show=True)
    
    def plot_training_progress(
        self,
        show: bool = True,
        save: bool = False
    ):
        """Plot current training progress."""
        fig, axs = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle('MPINN Training Progress', fontsize=14)
        
        # Plot losses
        self._plot_losses(axs[0, 0])
        
        # Plot alpha evolution
        self._plot_alpha_history(axs[0, 1])
        
        # Plot metrics if available
        if self.history['metrics']:
            self._plot_metrics(axs[1, 0])
        
        # Plot HF fraction performance if available
        if self.history['hf_losses']:
            self._plot_hf_performance(axs[1, 1])
        
        plt.tight_layout()
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / 'training_progress.png')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _plot_losses(self, ax):
        """Plot training losses."""
        if self.history['lf_loss']:
            ax.plot(self.history['lf_loss'], label='LF Loss')
        
        for fraction, losses in self.history['hf_losses'].items():
            ax.plot(losses, label=f'HF Loss ({fraction*100:.0f}%)')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
    
    def _plot_alpha_history(self, ax):
        """Plot alpha parameter evolution."""
        ax.plot(self.history['alpha_history'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Alpha')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True)
        ax.set_title('Alpha Parameter Evolution')
    
    def _plot_metrics(self, ax):
        """Plot training metrics."""
        for name, values in self.history['metrics'].items():
            ax.plot(values, label=name)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.legend()
        ax.grid(True)
    
    def _plot_hf_performance(self, ax):
        """Plot performance vs HF fraction."""
        fractions = sorted(self.history['hf_losses'].keys())
        final_losses = [min(losses) for losses in self.history['hf_losses'].values()]
        
        ax.plot(fractions, final_losses, 'o-')
        ax.set_xlabel('HF Data Fraction')
        ax.set_ylabel('Final Loss')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_title('Performance vs HF Fraction')
    
    def _display_progress(
        self,
        epoch: int,
        losses: Dict[str, float],
        alpha: float,
        hf_fraction: Optional[float]
    ):
        """Display current training progress."""
        phase = "LF" if hf_fraction is None else f"HF ({hf_fraction*100:.0f}%)"
        loss_str = f"Loss: {sum(losses.values()):.6f}"
        metrics_str = ""
        
        if self.history['metrics']:
            metrics_str = " | ".join(
                f"{name}: {values[-1]:.4f}"
                for name, values in self.history['metrics'].items()
            )
            metrics_str = f" | {metrics_str}"
        
        print(
            f"[{phase}] Epoch {epoch} | "
            f"{loss_str} | "
            f"Alpha: {alpha:.4f}"
            f"{metrics_str}"
        ) 