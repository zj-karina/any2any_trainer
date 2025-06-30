"""
Any2Any Trainer - Universal Multimodal Training Toolkit

Library for training any-to-any multimodal models.
"""

__version__ = "0.1.0"

# Core imports
from .models import (
    MultimodalModel,
    AnyToAnyModel,
    load_model,
)

from .utils import (
    ConfigManager,
    setup_logging,
    get_logger,
)

# Training imports
from .training import (
    MultimodalTrainer,
)

# Data imports  
from .data import (
    load_dataset,
    MultimodalCollator,
)

__all__ = [
    # Models
    "MultimodalModel",
    "AnyToAnyModel", 
    "load_model",
    # Utils
    "ConfigManager",
    "setup_logging",
    "get_logger",
    # Training
    "MultimodalTrainer",
    # Data
    "load_dataset",
    "MultimodalCollator",
] 