"""
Training package — Loss function dan trainer.

Ekspor utama:
- MultiTaskLoss
- Trainer
"""

from .losses import MultiTaskLoss
from .trainer import Trainer

__all__ = [
    "MultiTaskLoss",
    "Trainer",
]