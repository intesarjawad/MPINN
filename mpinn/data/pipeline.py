"""
Data pipeline orchestration for MPINN.

This module combines data loading and processing into a unified pipeline
for preparing molecular dynamics data for MPINN training.
"""

from pathlib import Path
from typing import List, Dict, Union, Optional

from .loader import DataLoader
from .processor import DataProcessor

class DataPipeline:
    """
    Orchestrates data loading and processing for MPINN training.
    
    This class provides a unified interface for:
    1. Loading HF/LF data
    2. Processing and standardizing data
    3. Managing data fractions
    4. Validating data consistency
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        input_features: List[str],
        output_features: List[str],
        hf_fractions: Optional[List[float]] = None
    ):
        """
        Initialize the DataPipeline.
        
        Args:
            data_dir: Path to data directory
            input_features: List of input feature names
            output_features: List of output feature names
            hf_fractions: List of HF data fractions to use (default: [1.0])
        """
        self.loader = DataLoader(data_dir, input_features, output_features)
        self.processor = DataProcessor()
        self.hf_fractions = hf_fractions or [1.0]
        
        # Validate fractions
        self._validate_fractions()
        
    def _validate_fractions(self) -> None:
        """Validate HF data fractions."""
        for fraction in self.hf_fractions:
            if not 0 < fraction <= 1:
                raise ValueError(
                    f"Invalid HF fraction {fraction}. Must be between 0 and 1"
                )
    
    def prepare_data(
        self,
        hf_fraction: float
    ) -> Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]:
        """
        Load and process data for a specific HF fraction.
        
        Args:
            hf_fraction: Fraction of HF data to use
            
        Returns:
            Processed data dictionary containing standardized HF and LF data
        """
        # Load raw data
        raw_data = self.loader.load_data(hf_fraction)
        
        # Fit processor if not already fitted
        if not self.processor._is_fitted:
            self.processor.fit(raw_data)
        
        # Process data
        processed_data = self.processor.transform(raw_data)
        
        return processed_data
    
    def get_all_fractions(
        self
    ) -> Dict[float, Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]]:
        """
        Load and process data for all specified HF fractions.
        
        Returns:
            Dictionary mapping HF fractions to processed data
        """
        return {
            fraction: self.prepare_data(fraction)
            for fraction in self.hf_fractions
        } 