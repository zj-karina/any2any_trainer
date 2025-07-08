#!/usr/bin/env python3
"""
Script for training multimodal models.

Supports various architectures (LLaVA-style, BLIP-style, etc.)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from transformers import set_seed

# Add src to PATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from any2any_trainer.utils import ConfigManager, setup_logging, get_logger
from any2any_trainer.models import load_model
from any2any_trainer.data import load_dataset, MultimodalCollator
from any2any_trainer.training import MultimodalTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multimodal models")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(
        level="DEBUG" if args.debug else "INFO",
        log_file=None,
        rich_console=True
    )
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting multimodal model training...")
    
    # Load configuration
    try:
        config = ConfigManager.load_config(args.config_path)
        logger.info(f"✅ Configuration loaded from {args.config_path}")
    except Exception as e:
        logger.error(f"❌ Error loading configuration: {e}")
        return 1
    
    # Validate configuration
    try:
        ConfigManager.validate_config(config)
    except Exception as e:
        logger.error(f"❌ Configuration validation error: {e}")
        return 1
    
    # Set seed
    if hasattr(config, 'seed'):
        set_seed(config.seed)
        logger.info(f"🌱 Seed set: {config.seed}")
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Log device information
    logger.info(f"🖥️  Device: {accelerator.device}")
    logger.info(f"🔢 Number of processes: {accelerator.num_processes}")
    
    # Load model
    try:
        logger.info("📥 Loading model...")
        model = load_model(config)
        logger.info("✅ Model loaded successfully")
        
        # Load tokenizer for data processing
        from any2any_trainer.models.factory import ModelFactory
        tokenizer = ModelFactory.load_tokenizer(config)
        logger.info("✅ Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"❌ Model loading error: {e}")
        return 1
    
    # Load data
    try:
        logger.info("📊 Loading data...")
        train_dataset, eval_dataset = load_dataset(config)
        logger.info(f"✅ Loaded {len(train_dataset)} training examples")
        if eval_dataset:
            logger.info(f"✅ Loaded {len(eval_dataset)} validation examples")
    except Exception as e:
        logger.error(f"❌ Data loading error: {e}")
        return 1
    
    # Create data collator
    try:
        data_collator = MultimodalCollator(config, tokenizer=tokenizer)
        logger.info("✅ Data collator created")
    except Exception as e:
        logger.error(f"❌ Data collator creation error: {e}")
        return 1
    
    # Create trainer
    try:
        logger.info("🏋️ Creating trainer...")
        trainer = MultimodalTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            accelerator=accelerator
        )
        logger.info("✅ Trainer created successfully")
    except Exception as e:
        logger.error(f"❌ Trainer creation error: {e}")
        return 1
    
    # Start training
    try:
        logger.info("🎯 Starting training...")
        trainer.train()
        logger.info("🎉 Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        return 1
    
    # Save model
    try:
        if accelerator.is_main_process:
            logger.info("💾 Saving final model...")
            output_dir = Path(config.output_dir) / "final_model"
            trainer.save_model(output_dir)
            logger.info(f"✅ Model saved to {output_dir}")
    except Exception as e:
        logger.error(f"❌ Model saving error: {e}")
        return 1
    
    logger.info("🏁 Training process completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 