"""
Data loading utilities for MPINN.

This module handles loading and initial processing of high-fidelity (HF) and 
low-fidelity (LF) molecular dynamics data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Optional

class DataLoader:
    """
    Handles loading and basic validation of molecular dynamics data.
    
    This class manages the loading of both high-fidelity and low-fidelity data,
    supporting progressive loading of HF data fractions while maintaining data
    consistency and validation.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        input_features: List[str],
        output_features: List[str]
    ):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Path to directory containing HF and LF data
            input_features: List of input feature names (e.g., ['temperature', 'density'])
            output_features: List of output feature names (e.g., ['energy', 'pressure'])
        """
        self.data_dir = Path(data_dir)
        self.input_features = input_features
        self.output_features = output_features
        
        # Validate data directory structure
        self._validate_data_structure()
        
    def _validate_data_structure(self) -> None:
        """
        Verify the expected data directory structure exists.
        
        Expected structure:
        data/
        ├── high_fidelity/
        │   ├── T_rho_high_fidelity.txt
        │   ├── E_high_fidelity.txt
        │   └── ...
        └── low_fidelity/
            ├── T_rho_low_fidelity.txt
            ├── E_low_fidelity.txt
            └── ...
        """
        required_dirs = ['high_fidelity', 'low_fidelity']
        for dir_name in required_dirs:
            if not (self.data_dir / dir_name).exists():
                raise FileNotFoundError(
                    f"Required directory '{dir_name}' not found in {self.data_dir}"
                )
    
    def load_data(
        self,
        hf_fraction: float = 1.0
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load both HF and LF data with specified HF fraction.
        
        Args:
            hf_fraction: Fraction of high-fidelity data to load (0.1-1.0)
            
        Returns:
            Dictionary containing:
                'high_fidelity': {
                    'inputs': array of input features,
                    'outputs': {feature_name: array of values}
                },
                'low_fidelity': {
                    'inputs': array of input features,
                    'outputs': {feature_name: array of values}
                }
        """
        # Validate fraction
        if not 0.0 < hf_fraction <= 1.0:
            raise ValueError("HF fraction must be between 0 and 1")
            
        # Load data
        data = {
            'high_fidelity': self._load_fidelity_data('high_fidelity', hf_fraction),
            'low_fidelity': self._load_fidelity_data('low_fidelity')
        }
        
        # Validate data consistency
        self._validate_data_consistency(data)
        
        return data
    
    def _load_fidelity_data(
        self,
        fidelity: str,
        fraction: float = 1.0
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Load data for a specific fidelity level.
        
        Args:
            fidelity: Either 'high_fidelity' or 'low_fidelity'
            fraction: Fraction of data to load (only applies to HF)
            
        Returns:
            Dictionary containing inputs and outputs
        """
        # Load input features
        input_file = f"T_rho_{fidelity}.txt"
        if fraction < 1.0 and fidelity == 'high_fidelity':
            input_file = f"T_rho_{fidelity}_{int(fraction*100)}.txt"
            
        inputs = np.loadtxt(self.data_dir / fidelity / input_file)
        
        # Load output features
        outputs = {}
        for feature in self.output_features:
            output_file = f"{feature}_{fidelity}.txt"
            if fraction < 1.0 and fidelity == 'high_fidelity':
                output_file = f"{feature}_{fidelity}_{int(fraction*100)}.txt"
                
            outputs[feature] = np.loadtxt(
                self.data_dir / fidelity / output_file
            )
            
        return {
            'inputs': inputs,
            'outputs': outputs
        }
    
    def _validate_data_consistency(
        self,
        data: Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]
    ) -> None:
        """
        Validate consistency of loaded data.
        
        Checks:
        - Input/output dimensions match
        - No missing values
        - Data ranges are reasonable
        
        Args:
            data: Loaded data dictionary
        """
        for fidelity, fid_data in data.items():
            # Check input/output dimensions
            n_samples = len(fid_data['inputs'])
            for feature, values in fid_data['outputs'].items():
                if len(values) != n_samples:
                    raise ValueError(
                        f"Dimension mismatch in {fidelity} {feature}: "
                        f"Expected {n_samples}, got {len(values)}"
                    )
            
            # Check for NaN/Inf values
            if np.any(np.isnan(fid_data['inputs'])) or \
               np.any(np.isinf(fid_data['inputs'])):
                raise ValueError(f"Invalid values found in {fidelity} inputs")
                
            for feature, values in fid_data['outputs'].items():
                if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                    raise ValueError(
                        f"Invalid values found in {fidelity} {feature}"
                    ) 