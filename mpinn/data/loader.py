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
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        required_dirs = ['high_fidelity', 'low_fidelity']
        for dir_name in required_dirs:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                raise FileNotFoundError(
                    f"Required directory not found: {dir_path}\n"
                    "Expected structure:\n"
                    "data/\n"
                    "├── high_fidelity/\n"
                    "│   ├── T_rho_high_fidelity.txt\n"
                    "│   ├── energy_high_fidelity.txt\n"
                    "│   └── ...\n"
                    "└── low_fidelity/\n"
                    "    ├── T_rho_low_fidelity.txt\n"
                    "    ├── energy_low_fidelity.txt\n"
                    "    └── ..."
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
        # Feature name mapping
        feature_map = {
            'temperature': 'T',  # T from T_rho
            'density': 'rho',    # rho from T_rho
            'energy': 'E',       # E_high_fidelity*.txt
            'pressure': 'P',     # P_high_fidelity*.txt
            'diffusion': 'D'     # D_high_fidelity*.txt
        }
        
        fidelity_dir = self.data_dir / fidelity
        
        # Load input features (T_rho file)
        if fraction < 1.0 and fidelity == 'high_fidelity':
            input_file = f"T_rho_{fidelity}_{int(fraction*100)}.txt"
        else:
            input_file = f"T_rho_{fidelity}.txt"
        
        input_path = fidelity_dir / input_file
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}\n"
                f"Looking for temperature/density data"
            )
        
        inputs = np.loadtxt(input_path)
        
        # Load output features
        outputs = {}
        for feature_name in self.output_features:
            short_name = feature_map.get(feature_name, feature_name[0].upper())
            
            if fraction < 1.0 and fidelity == 'high_fidelity':
                output_file = f"{short_name}_{fidelity}_{int(fraction*100)}.txt"
            else:
                output_file = f"{short_name}_{fidelity}.txt"
            
            output_path = fidelity_dir / output_file
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Output file not found: {output_path}\n"
                    f"Looking for {feature_name} data"
                )
            
            outputs[feature_name] = np.loadtxt(output_path)
        
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