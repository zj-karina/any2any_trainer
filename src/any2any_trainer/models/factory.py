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
    
    else:
        # Simple loading of regular LLM
        model = ModelFactory.load_base_model(config)
        
        # Set up PEFT
        model = ModelFactory.setup_peft(model, config)
        
        # Freeze parameters
        ModelFactory.freeze_parameters(model, config)
        
        return model 