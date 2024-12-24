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
        # Compute input stats from both HF and LF data
        all_inputs = []
        for fidelity in ['high_fidelity', 'low_fidelity']:
            if fidelity in data:
                all_inputs.append(data[fidelity]['inputs'])
        
        self.stats = {
            'inputs': self._compute_stats(np.vstack(all_inputs))
        }
        
        # Compute output stats for each feature
        self.stats['outputs'] = {}
        for feature in data['high_fidelity']['outputs'].keys():
            all_outputs = []
            for fidelity in ['high_fidelity', 'low_fidelity']:
                if fidelity in data and feature in data[fidelity]['outputs']:
                    all_outputs.append(data[fidelity]['outputs'][feature])
            
            self.stats['outputs'][feature] = self._compute_stats(
                np.concatenate(all_outputs)
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
        # Ensure stats match data shape for broadcasting
        if len(stats['mean'].shape) > 1:
            stats['mean'] = stats['mean'].reshape(-1)
        if len(stats['std'].shape) > 1:
            stats['std'] = stats['std'].reshape(-1)
        
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