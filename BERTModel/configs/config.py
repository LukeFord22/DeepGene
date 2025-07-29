"""Default configuration for DNA SNP detection training."""

import os
from dataclasses import dataclass, field
from datetime import datetime

os.environ['WANDB_API_KEY'] = 'f9ef5f01cec347e6598e41294b2dce5380dedbc1'

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 32
    num_epochs: int = 12
    learning_rate: float = 3e-5
    grad_accum_steps: int = 2
    train_data_fraction: float = 1
    warmup_steps: int = 100
    max_token_length: int = 512
    random_seed: int = 42  
    

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    num_attention_heads: int = 16

@dataclass
class DataConfig:
    """Data configuration parameters."""
    data_dir: str = "data/SNPdata"

    # .pt file configuration 
    pt_file_pattern: str = "_tokenized_batch_*.pt"  # Pattern to match .pt batch files
    pt_cache_size: int = 3  # Number of .pt files to keep in memory cache

    # Data loading settings
    test_fraction: float = 0.1
    pin_memory: bool = True  # Faster GPU transfer
    persistent_workers: bool = True  # Keep workers alive between epochs

@dataclass
class DeepSpeedConfig:
    """DeepSpeed configuration parameters."""
    zero_stage: int = 1
    fp16_enabled: bool = False
    offload_optimizer: bool = True
    gradient_clipping: float = 5.0


@dataclass
class Config:
    """Combined configuration for the entire project."""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    
    # Experiment settings
    experiment_name: str = field(default_factory=lambda: f"dna-snp-detection-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    project_name: str = "dna-snp-detection-deepspeed"
    output_dir: str = "checkpoints"
    
    # Device settings
    local_rank: int = -1
    device: str = "cuda"

# Create default configuration instance
default_config = Config()