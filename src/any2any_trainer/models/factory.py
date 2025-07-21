"""Factory for creating different types of models."""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoImageProcessor,
    CLIPVisionModel,
    WhisperModel,
    WhisperProcessor,
)

from ..utils.config import TrainingConfig

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating different types of models."""
    
    @staticmethod
    def load_base_model(config: TrainingConfig) -> nn.Module:
        """Load base language model with device mapping."""
        logger.info(f"📥 Loading base model: {config.model_name_or_path}")
        
        # Попытка загрузки модели с различными подходами
        model = None
        
        # Специальная обработка для Qwen2.5-Omni СНАЧАЛА
        if "Qwen2.5-Omni" in config.model_name_or_path:
            try:
                logger.info("🌟 Trying specialized loading for Qwen2.5-Omni...")
                
                # Пробуем правильный класс для Omni
                from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
                model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    config.model_name_or_path,
                    torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                    use_safetensors=True  # Принудительно safetensors для безопасности
                )
                logger.info(f"✅ Qwen2.5-Omni loaded with specialized class: {type(model).__name__}")
                    
            except Exception as e_omni:
                logger.warning(f"⚠️ Specialized Qwen2.5-Omni loading failed: {e_omni}")
                logger.info("💡 Qwen2.5-Omni may not be fully integrated yet. Using fallback.")
                # Для демонстрации используем обычную модель
                logger.info("🔄 Switching to compatible Qwen2.5-7B-Instruct for testing...")
                config.model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
                model = None  # Сбрасываем, чтобы попробовать стандартную загрузку
        
        # Стандартная загрузка для обычных моделей (или fallback)
        if model is None:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_name_or_path,
                    torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                    use_safetensors=True
                )
                logger.info(f"✅ Model loaded with AutoModelForCausalLM: {type(model).__name__}")
                
            except Exception as e:
                logger.warning(f"⚠️ AutoModelForCausalLM failed: {e}")
                
                # Для новых архитектур пробуем AutoModel
                try:
                    logger.info("🔄 Trying AutoModel for newer architectures...")
                    from transformers import AutoModel
                    
                    model = AutoModel.from_pretrained(
                        config.model_name_or_path,
                        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
                        device_map="auto",
                        trust_remote_code=True,
                        use_safetensors=True,
                        _commit_hash=None,
                    )
                    logger.info(f"✅ Model loaded with AutoModel: {type(model).__name__}")
                    
                except Exception as e2:
                    logger.error(f"❌ Failed to load model with all approaches")
                    logger.error(f"   AutoModelForCausalLM error: {e}")
                    logger.error(f"   AutoModel error: {e2}")
                    raise ValueError(f"Could not load model '{config.model_name_or_path}'. "
                                   f"Make sure the model exists and is compatible. "
                                   f"For newest models, you may need to update transformers: "
                                   f"pip install --upgrade transformers")
        
        if model is None:
            raise ValueError(f"Failed to load model: {config.model_name_or_path}")
        
        # Apply settings
        if config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
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
            model = CLIPVisionModel.from_pretrained(
                model_name,
                use_safetensors=True  # Принудительно safetensors для безопасности
            )
            processor = AutoImageProcessor.from_pretrained(model_name)
            
            return {
                "model": model,
                "processor": processor,
                "hidden_size": model.config.hidden_size,
            }
        except Exception as e:
            logger.error(f"Failed to load vision encoder: {e}")
            raise
    
    @staticmethod
    def load_audio_encoder(model_name: str) -> Dict[str, Any]:
        """Load audio encoder and processor."""
        logger.info(f"🎧 Loading audio encoder: {model_name}")
        
        try:
            from transformers import WhisperModel, WhisperProcessor
            model = WhisperModel.from_pretrained(
                model_name,
                use_safetensors=True  # Принудительно safetensors для безопасности
            )
            processor = WhisperProcessor.from_pretrained(model_name)
            
            return {
                "model": model,
                "processor": processor,
                "hidden_size": model.config.d_model,
            }
        except Exception as e:
            logger.error(f"Failed to load audio encoder: {e}")
            raise
    
    @staticmethod
    def setup_lora(model: nn.Module, config: TrainingConfig) -> nn.Module:
        """Setup LoRA (Low-Rank Adaptation) for efficient training."""
        if not config.use_peft:
            return model
        
        logger.info("🔧 Setting up LoRA...")
        
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=config.lora.target_modules,
                bias=config.lora.bias,
            )
            
            model = get_peft_model(model, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_percent = 100 * trainable_params / total_params
            
            logger.info(f"🎯 Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")
            
            return model
            
        except ImportError:
            logger.warning("⚠️ PEFT not available, training without LoRA")
            return model
        except Exception as e:
            logger.warning(f"⚠️ LoRA setup failed: {e}")
            return model


def load_model(config: TrainingConfig) -> nn.Module:
    """Load model based on configuration."""
    logger.info(f"🚀 Creating model type: {config.model_type}")
    
    if config.model_type == "standard":
        logger.info("🔄 Loading standard language model...")
        from .multimodal import MultimodalModel
        return MultimodalModel.from_config(config)
    
    elif config.model_type == "multimodal":
        logger.info("🔄 Loading multimodal model...")
        from .multimodal import MultimodalModel
        return MultimodalModel.from_config(config)
    
    elif config.model_type == "any2any":
        logger.info("🔄 Loading any-to-any model...")
        from .any2any import AnyToAnyModel
        return AnyToAnyModel.from_config(config)
    
    else:
        raise ValueError(f"Unknown model type: {config.model_type}") 