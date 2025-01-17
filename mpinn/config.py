"""
Configuration management for MPINN.
"""

import yaml
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class MPINNConfig:
    """Configuration for MPINN training."""
    
    # Data configuration
    data_dir: Path
    input_features: List[str]
    output_features: List[str]
    hf_fractions: List[float]
    
    # Model architecture
    hidden_layers_lf: List[int] = field(default_factory=lambda: [20, 20, 20, 20])
    hidden_layers_hf: List[int] = field(default_factory=lambda: [20, 20])
    activation: str = 'tanh'
    l2_reg: float = 0.001
    
    # Training parameters
    lf_epochs: int = 1000
    hf_epochs: int = 500
    batch_size: int = 32
    lf_patience: int = 50
    hf_patience: int = 30
    
    # Optimizer settings
    optimizer_config: Dict = field(
        default_factory=lambda: {
            'lf': {'optimizer': 'Adam', 'learning_rate': 0.001},
            'hf': {'optimizer': 'Adam', 'learning_rate': 0.0005}
        }
    )
    
    # Output settings
    save_dir: Optional[Path] = None
    log_frequency: int = 10 
    
    def save(self, path: Path):
        """Save configuration to YAML file."""
        config_dict = asdict(self)
        
        # Convert Path objects to strings
        config_dict['data_dir'] = str(config_dict['data_dir'])
        if config_dict['save_dir']:
            config_dict['save_dir'] = str(config_dict['save_dir'])
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Path) -> 'MPINNConfig':
        """Load configuration from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        
        # Convert string paths back to Path objects
        config_dict['data_dir'] = Path(config_dict['data_dir'])
        if config_dict['save_dir']:
            config_dict['save_dir'] = Path(config_dict['save_dir'])
        
        return cls(**config_dict) 
    
    def validate(self):
        """Validate configuration settings."""
        # Validate data directory
        if not self.data_dir.exists():
            raise ValueError(
                f"Data directory does not exist: {self.data_dir}\n"
                "Please ensure the correct data path is provided."
            )
        
        # Validate features
        if not self.input_features:
            raise ValueError("No input features specified")
        if not self.output_features:
            raise ValueError("No output features specified")
        
        # Validate HF fractions
        if not all(0 < f <= 1 for f in self.hf_fractions):
            raise ValueError("HF fractions must be between 0 and 1")
        if 1.0 not in self.hf_fractions:
            self.hf_fractions.append(1.0)
            print("Added 1.0 to hf_fractions for final evaluation")
        
        # Validate network architecture
        if not self.hidden_layers_lf or not self.hidden_layers_hf:
            raise ValueError("Network architecture not properly specified")
        
        # Validate optimizer settings
        required_opt_keys = {'optimizer', 'learning_rate'}
        for phase in ['lf', 'hf']:
            if not all(k in self.optimizer_config[phase] for k in required_opt_keys):
                raise ValueError(f"Missing required optimizer settings for {phase}") 