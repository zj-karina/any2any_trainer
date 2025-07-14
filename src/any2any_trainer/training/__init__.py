"""
Training module for Any2Any Trainer.

Provides training functionality for multimodal models.
"""

from .trainer import MultimodalTrainer, SimpleTrainer

__all__ = [
    "MultimodalTrainer",
    "SimpleTrainer",
] 