"""
Model factory for Any2Any Trainer.

Simple approach with direct use of HuggingFace models.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoConfig,
)
from peft import get_peft_model, LoraConfig, TaskType

from ..utils.config import TrainingConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """Factory for creating models."""
    
    @staticmethod
    def load_base_model(config: TrainingConfig) -> nn.Module:
        """Load base model from HuggingFace."""
        logger.info(f"📥 Loading base model: {config.model_name_or_path}")
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
            device_map="auto" if torch.cuda.device_count() > 1 else None,
            trust_remote_code=True,
        )
        
        return model
    
    @staticmethod
    def load_tokenizer(config: TrainingConfig) -> Any:
        """Load tokenizer."""
        logger.info(f"📝 Loading tokenizer: {config.model_name_or_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
        )
        
        # Add special tokens if they exist
        special_tokens = []
        for token_name, token_value in config.special_tokens.items():
            special_tokens.append(token_value)
        
        if special_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        # Set pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    
    @staticmethod
    def load_vision_encoder(model_name: str) -> Dict[str, Any]:
        """Load vision encoder."""
        logger.info(f"👁️ Loading vision encoder: {model_name}")
        
        try:
            # Try loading with safetensors first to avoid torch.load security issues
            model = AutoModel.from_pretrained(
                model_name, 
                trust_remote_code=True,
                use_safetensors=True  # Force safetensors usage
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to load with safetensors, trying default: {e}")
            # Fallback to default loading
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        return {
            "model": model.vision_model if hasattr(model, 'vision_model') else model,
            "processor": processor,
            "config": model.config,
        }
    
    @staticmethod
    def load_audio_encoder(model_name: str) -> Dict[str, Any]:
        """Load audio encoder."""
        logger.info(f"🎵 Loading audio encoder: {model_name}")
        
        try:
            # Try loading with safetensors first to avoid torch.load security issues
            model = AutoModel.from_pretrained(
                model_name, 
                trust_remote_code=True,
                use_safetensors=True  # Force safetensors usage
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to load with safetensors, trying default: {e}")
            # Fallback to default loading
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        return {
            "model": model.encoder if hasattr(model, 'encoder') else model,
            "processor": processor,
            "config": model.config,
        }
    
    @staticmethod
    def setup_peft(model: nn.Module, config: TrainingConfig) -> nn.Module:
        """Set up PEFT (LoRA) for the model."""
        if not config.use_peft:
            return model
            
        logger.info("🔧 Setting up LoRA...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            bias=config.lora.bias,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Explicitly enable gradients for LoRA parameters
        trainable_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_count += 1
                if trainable_count <= 5:  # Log first 5 only
                    logger.info(f"✅ Trainable parameter: {name}")
        
        logger.info(f"📊 Total trainable parameters: {trainable_count}")
        
        # Force enable training mode and ensure LoRA is active
        model.train()
        
        # Double-check LoRA is working
        if hasattr(model, 'peft_config'):
            logger.info("✅ PEFT/LoRA is active")
        
        # Force gradients on
        for name, param in model.named_parameters():
            if 'lora' in name.lower() or 'adapters' in name.lower():
                param.requires_grad_(True)
                logger.debug(f"🔧 Forced gradients for LoRA param: {name}")
        
        return model
    
    @staticmethod
    def freeze_parameters(model: nn.Module, config: TrainingConfig) -> None:
        """Freeze model parameters according to configuration."""
        if config.train_projection_only:
            # Freeze everything except projection layers
            for name, param in model.named_parameters():
                if "projector" not in name.lower() and "projection" not in name.lower():
                    param.requires_grad = False
                    
        # Unfreeze specific layers by patterns
        for pattern in config.unfreeze_layers_patterns:
            for name, param in model.named_parameters():
                if pattern in name:
                    param.requires_grad = True
                    logger.info(f"🔓 Unfrozen layer: {name}")
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🎯 Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Ensure we have at least some trainable parameters
        if trainable_params == 0:
            logger.warning("⚠️ No trainable parameters found! This will cause training errors.")
            # Force enable some parameters for LoRA
            if hasattr(model, 'base_model'):
                for name, param in model.base_model.named_parameters():
                    if 'lora' in name.lower():
                        param.requires_grad = True
                        break


def load_model(config: TrainingConfig) -> nn.Module:
    """
    Load model according to configuration.
    
    Simple approach like align-anything - use HF models directly.
    """
    logger.info(f"🚀 Creating model type: {config.model_type}")
    
    if config.model_type == "multimodal":
        from .multimodal import MultimodalModel
        return MultimodalModel.from_config(config)
    
    elif config.model_type == "any2any":
        from .any2any import AnyToAnyModel  
        return AnyToAnyModel.from_config(config)
    
    elif config.model_type == "standard" or config.model_type == "auto":
        # Standard HuggingFace model loading
        logger.info("📚 Loading standard HuggingFace model...")
        model = ModelFactory.load_base_model(config)
        
        # Set up PEFT
        model = ModelFactory.setup_peft(model, config)
        
        # Freeze parameters
        ModelFactory.freeze_parameters(model, config)
        
        return model
    
    else:
        # Fallback for unknown types - treat as standard
        logger.warning(f"⚠️ Unknown model type '{config.model_type}', treating as standard")
        model = ModelFactory.load_base_model(config)
        
        # Set up PEFT
        model = ModelFactory.setup_peft(model, config)
        
        # Freeze parameters
        ModelFactory.freeze_parameters(model, config)
        
        return model 