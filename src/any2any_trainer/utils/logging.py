"""
Logging system for Any2Any Trainer.
"""

import logging
import sys
from typing import Optional
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rich_console: bool = True
) -> None:
    """Set up logging system."""
    
    # Set logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # List of handlers
    handlers = []
    
    # Console handler with Rich
    if rich_console:
        console = Console()
        rich_handler = RichHandler(
            console=console,
            show_path=False,
            show_time=False,
            rich_tracebacks=True
        )
        rich_handler.setLevel(log_level)
        handlers.append(rich_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )
    
    # Configure loggers for external libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get logger with given name."""
    return logging.getLogger(name) 