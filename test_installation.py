#!/usr/bin/env python3
"""
Test script to validate Any2Any Trainer installation and basic functionality.

Run this script to check if the library works out of the box.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test that all main modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test core imports
        from any2any_trainer import (
            load_model, MultimodalTrainer, MultimodalCollator,
            load_dataset, ConfigManager, setup_logging, get_logger
        )
        print("✅ Core imports successful")
        
        # Test configuration loading
        from any2any_trainer.utils.config import TrainingConfig
        print("✅ Configuration system available")
        
        # Test data components
        from any2any_trainer.data import load_dataset as data_load_dataset
        from any2any_trainer.data.collator import MultimodalCollator as DataCollator
        print("✅ Data components available")
        
        # Test training components
        from any2any_trainer.training import MultimodalTrainer as Trainer
        print("✅ Training components available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\n🔧 Testing configuration loading...")
    
    try:
        from any2any_trainer.utils.config import ConfigManager
        
        # Test if we can create a basic config
        config_data = {
            "model_name": "gpt2",
            "output_dir": "test_output",
            "dataset": ["wikitext", "wikitext-2-raw-v1"],
            "learning_rate": 5e-5,
            "batch_size": 2,
            "max_epochs": 1
        }
        
        config = ConfigManager.from_dict(config_data)
        print(f"✅ Config loaded: model={config.model_name}, lr={config.learning_rate}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_model_loading():
    """Test basic model loading functionality."""
    print("\n🤖 Testing model loading...")
    
    try:
        from any2any_trainer.models import load_model
        from any2any_trainer.utils.config import ConfigManager
        
        # Simple config for model loading
        config_data = {
            "model_name": "gpt2",
            "output_dir": "test_output",
            "dataset": ["wikitext"],
            "use_peft": True,
            "lora_r": 8
        }
        
        config = ConfigManager.from_dict(config_data)
        
        # This might take a moment to download the model
        print("⏳ Loading GPT2 model (this may take a moment)...")
        model, tokenizer = load_model(config)
        
        print(f"✅ Model loaded successfully: {type(model).__name__}")
        print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")
        
        # Test a simple forward pass
        print("🧪 Testing model forward pass...")
        import torch
        
        # Create dummy input
        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Forward pass successful, output shape: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("ℹ️  This might be due to network issues or missing dependencies")
        return False


def test_logging():
    """Test logging system."""
    print("\n📝 Testing logging system...")
    
    try:
        from any2any_trainer.utils.logging import setup_logging, get_logger
        
        # Setup logging
        setup_logging(level="INFO")
        logger = get_logger("test")
        
        logger.info("Test log message")
        print("✅ Logging system working")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Any2Any Trainer Installation Test")
    print("=" * 40)
    
    all_passed = True
    
    # Run tests
    tests = [
        ("Import test", test_imports),
        ("Configuration test", test_config_loading),
        ("Logging test", test_logging),
        ("Model loading test", test_model_loading),  # This one might be slow
    ]
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Any2Any Trainer is ready to use.")
        print("\n📚 Next steps:")
        print("1. Try running: python scripts/train_multimodal.py --config configs/test/minimal_test.yaml")
        print("2. Check out the examples in configs/ directory")
        print("3. Read the documentation in README.md and QUICKSTART.md")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -e .")
        print("2. Check if you have sufficient disk space for model downloads")
        print("3. Verify your internet connection for downloading models")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 