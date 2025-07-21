"""
Data module for Any2Any Trainer.

Provides data loading and processing functionality.
"""

from .dataset import load_dataset
from .collator import MultimodalCollator

__all__ = [
    "load_dataset",
    "MultimodalCollator",
] 