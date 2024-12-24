"""
Data processing utilities for MPINN.

This module handles data standardization, transformation, and preparation
for training the MPINN model.
"""

import numpy as np
from typing import Dict, Union, Optional, Tuple

class DataProcessor:
    """
    Handles data preprocessing and standardization for MPINN training.
    
    Features:
    - Data standardization (zero mean, unit variance)
    - Feature scaling
    - Statistics tracking for denormalization
    """
    
    def __init__(self):
        """Initialize the DataProcessor with empty statistics."""
        self.stats = {}
        self._is_fitted = False
        
    def fit(
        self,
        data: Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]
    ) -> None:
        """
        Compute standardization parameters from data.
        
        Args:
            data: Dictionary containing HF and LF data
        """
        self.stats = {
            'inputs': self._compute_stats(
                np.vstack([
                    data['high_fidelity']['inputs'],
                    data['low_fidelity']['inputs']
                ])
            )
        }
        
        # Compute stats for each output feature
        self.stats['outputs'] = {}
        for feature in data['high_fidelity']['outputs'].keys():
            self.stats['outputs'][feature] = self._compute_stats(
                np.vstack([
                    data['high_fidelity']['outputs'][feature],
                    data['low_fidelity']['outputs'][feature]
                ])
            )
            
        self._is_fitted = True
        
    def transform(
        self,
        data: Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]
    ) -> Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]:
        """
        Standardize data using computed statistics.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Standardized data dictionary
        """
        if not self._is_fitted:
            raise RuntimeError("DataProcessor must be fitted before transform")
            
        processed = {}
        for fidelity, fid_data in data.items():
            processed[fidelity] = {
                'inputs': self._standardize(
                    fid_data['inputs'],
                    self.stats['inputs']
                ),
                'outputs': {}
            }
            
            for feature, values in fid_data['outputs'].items():
                processed[fidelity]['outputs'][feature] = self._standardize(
                    values,
                    self.stats['outputs'][feature]
                )
                
        return processed
    
    def inverse_transform(
        self,
        data: Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]
    ) -> Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]:
        """
        Convert standardized data back to original scale.
        
        Args:
            data: Standardized data dictionary
            
        Returns:
            Data dictionary in original scale
        """
        if not self._is_fitted:
            raise RuntimeError("DataProcessor must be fitted before inverse_transform")
            
        original = {}
        for fidelity, fid_data in data.items():
            original[fidelity] = {
                'inputs': self._destandardize(
                    fid_data['inputs'],
                    self.stats['inputs']
                ),
                'outputs': {}
            }
            
            for feature, values in fid_data['outputs'].items():
                original[fidelity]['outputs'][feature] = self._destandardize(
                    values,
                    self.stats['outputs'][feature]
                )
                
        return original
    
    @staticmethod
    def _compute_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute mean and standard deviation for standardization.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary with 'mean' and 'std' arrays
        """
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0)
        }
    
    @staticmethod
    def _standardize(
        data: np.ndarray,
        stats: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Standardize data to zero mean and unit variance.
        
        Args:
            data: Input data array
            stats: Dictionary with 'mean' and 'std' arrays
            
        Returns:
            Standardized data array
        """
        return (data - stats['mean']) / (stats['std'] + 1e-8)
    
    @staticmethod
    def _destandardize(
        data: np.ndarray,
        stats: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Convert standardized data back to original scale.
        
        Args:
            data: Standardized data array
            stats: Dictionary with 'mean' and 'std' arrays
            
        Returns:
            Data array in original scale
        """
        return data * stats['std'] + stats['mean'] 