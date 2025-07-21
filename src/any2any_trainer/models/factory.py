"""
Model factory for loading different types of models.
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoImageProcessor,
    AutoProcessor,
    BitsAndBytesConfig
)

from ..utils.config import TrainingConfig
from .multimodal import MultimodalModel
from .any2any import AnyToAnyModel

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating different types of models."""
    
    @staticmethod
    def load_base_model(config: TrainingConfig) -> nn.Module:
        """Load base language model with device mapping."""
        logger.info(f"📥 Loading base model: {config.model_name_or_path}")
        
        # Load model with auto device mapping for GPU
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
            device_map="auto",  # Automatically place on GPU
            trust_remote_code=True,
            use_safetensors=True  # Принудительно используем safetensors для безопасности
        )
        
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model
    
    @staticmethod
    def load_tokenizer(config: TrainingConfig) -> Any:
        """Load tokenizer for the model."""
        logger.info(f"🔤 Loading tokenizer: {config.model_name_or_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add special tokens if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add custom special tokens
        if config.special_tokens:
            special_tokens_dict = {"additional_special_tokens": list(config.special_tokens.values())}
            tokenizer.add_special_tokens(special_tokens_dict)
        
        return tokenizer
    
    @staticmethod
    def load_vision_encoder(model_name: str) -> Dict[str, Any]:
        """Load vision encoder and processor."""
        logger.info(f"👁️ Loading vision encoder: {model_name}")
        
        try:
            from transformers import CLIPVisionModel
            model = CLIPVisionModel.from_pretrained(model_name)
            processor = AutoImageProcessor.from_pretrained(model_name)
            
            return {
                "model": model,
                "processor": processor,
                "hidden_size": model.config.hidden_size
            }
        except Exception as e:
            logger.error(f"Failed to load vision encoder: {e}")
            raise
    
    @staticmethod
    def load_audio_encoder(model_name: str) -> Dict[str, Any]:
        """Load audio encoder and processor."""
        logger.info(f"🎵 Loading audio encoder: {model_name}")
        
        try:
            from transformers import WhisperModel
            model = WhisperModel.from_pretrained(model_name)
            processor = AutoProcessor.from_pretrained(model_name)
            
            return {
                "model": model.encoder,  # Use only encoder part
                "processor": processor,
                "hidden_size": model.config.d_model
            }
        except Exception as e:
            logger.error(f"Failed to load audio encoder: {e}")
            raise
    
    @staticmethod
    def setup_peft(model: nn.Module, config: TrainingConfig) -> nn.Module:
        """Setup PEFT (LoRA) for the model with fallback to full training."""
        if not config.use_peft:
            logger.info("📚 Using full model training (PEFT disabled)")
            return model
        
        logger.info("🔧 Setting up LoRA...")
        
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            
            # Check if bitsandbytes is available
            try:
                import bitsandbytes
                logger.info("✅ Bitsandbytes available for quantization")
                use_quantization = True
            except ImportError:
                logger.info("⚠️ Bitsandbytes not available, using standard LoRA")
                use_quantization = False
            
            # Simple LoRA setup without quantization
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=config.lora.target_modules or ["q_proj", "v_proj"],
                bias=config.lora.bias,
            )
            
            model = get_peft_model(model, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"🎯 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
            
            return model
            
        except Exception as e:
            logger.warning(f"⚠️ LoRA setup failed: {e}")
            logger.info("🔄 Falling back to full model training")
            return model
    
    @staticmethod
    def freeze_parameters(model: nn.Module, config: TrainingConfig) -> None:
        """Freeze specific parameters based on configuration."""
        if config.train_projection_only:
            # Freeze everything except projection layers
            for name, param in model.named_parameters():
                if "projection" not in name.lower():
                    param.requires_grad = False
            logger.info("🧊 Frozen all parameters except projection layers")
            
        elif config.unfreeze_layers_patterns:
            # Freeze all parameters first
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze specific patterns
            for name, param in model.named_parameters():
                for pattern in config.unfreeze_layers_patterns:
                    if pattern in name:
                        param.requires_grad = True
                        break
            
            unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"🔓 Unfrozen {unfrozen_params:,} parameters ({100 * unfrozen_params / total_params:.2f}%)")


def load_model(config: TrainingConfig) -> nn.Module:
    """Main function to load model based on configuration."""
    logger.info(f"🚀 Creating model type: {config.model_type}")
    
    if config.model_type == "standard":
        logger.info("📚 Loading standard HuggingFace model...")
        
        # Load base model
        model = ModelFactory.load_base_model(config)
        
        # Setup PEFT if requested
        if config.use_peft:
            model = ModelFactory.setup_peft(model, config)
        
        # Apply freezing if needed
        ModelFactory.freeze_parameters(model, config)
        
        return model
        
    elif config.model_type == "multimodal":
        logger.info("🖼️ Loading multimodal model...")
        return MultimodalModel.from_config(config)
        
    elif config.model_type == "any2any":
        logger.info("🔄 Loading any-to-any model...")
        return AnyToAnyModel.from_config(config)
        
    else:
        raise ValueError(f"Unknown model type: {config.model_type}") 