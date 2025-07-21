#!/usr/bin/env python3
"""
Simple training script for any2any multimodal models.
"""

import sys
import os
import argparse
import logging
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from any2any_trainer.utils.config import ConfigManager, TrainingConfig
from any2any_trainer.utils.logging import setup_logging, get_logger
# Временное решение проблемы с импортом factory
import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from any2any_trainer.models.factory import load_model
from any2any_trainer.data.dataset import load_dataset
from any2any_trainer.data.collator import MultimodalCollator
from any2any_trainer.training.trainer import SimpleTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train multimodal models")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only test model loading and setup without training"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(level="INFO", rich_console=True)
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting multimodal model training...")
    
    # Load and validate configuration
    try:
        config = ConfigManager.load_config(args.config_path)
        logger.info(f"✅ Configuration loaded from {args.config_path}")
        
        # Validate configuration
        ConfigManager.validate_config(config)
        print("✅ Configuration passed validation")
        logger.info("✅ Configuration passed validation")
        
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return 1
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"🔢 Number of GPUs: {torch.cuda.device_count()}")
    
    try:
        # Load model
        logger.info("📥 Loading model...")
        model = load_model(config)
        logger.info("✅ Model loaded successfully")
        
        # Load tokenizer 
        from any2any_trainer.models.factory import ModelFactory
        tokenizer = ModelFactory.load_tokenizer(config)
        logger.info("✅ Tokenizer loaded successfully")
        
        # Load datasets
        logger.info("📊 Loading datasets...")
        train_dataset, eval_dataset = load_dataset(config)
        logger.info(f"✅ Train dataset: {len(train_dataset)} examples")
        if eval_dataset:
            logger.info(f"✅ Eval dataset: {len(eval_dataset)} examples")
        
        if args.dry_run:
            logger.info("🏁 Dry run completed successfully - stopping before training")
            return 0
        
        # Create data collator
        collator = MultimodalCollator(config, tokenizer)
        
        # Create data loaders
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        # Use SimpleTrainer
        trainer = SimpleTrainer(model, tokenizer, config)
        
        # Train model
        logger.info("🎯 Starting training...")
        trainer.train(train_dataloader, config.num_train_epochs)
        
        # Save model
        logger.info(f"💾 Saving model to {config.output_dir}")
        trainer.save_model(config.output_dir)
        
        logger.info("🎉 Training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main()) 