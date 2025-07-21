"""
Multimodal model (LLaVA-style).

Simple approach with direct use of HuggingFace models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import will be done locally to avoid circular imports
from ..utils.config import TrainingConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ProjectionLayer(nn.Module):
    """Simple projection layer for transforming vision embeddings to text space."""
    
    def __init__(self, vision_hidden_size: int, text_hidden_size: int, projection_type: str = "mlp"):
        super().__init__()
        
        if projection_type == "linear":
            self.projection = nn.Linear(vision_hidden_size, text_hidden_size)
        elif projection_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(vision_hidden_size, text_hidden_size),
                nn.ReLU(),
                nn.Linear(text_hidden_size, text_hidden_size)
            )
        else:
            raise ValueError(f"Unsupported projection type: {projection_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class MultimodalModel(nn.Module):
    """
    Multimodal model in LLaVA style.
    
    Uses standard HuggingFace models directly.
    """
    
    def __init__(
        self,
        language_model: nn.Module,
        vision_encoder: Dict[str, Any],
        projection_layer: nn.Module,
        tokenizer: Any,
        config: TrainingConfig,
    ):
        super().__init__()
        
        self.language_model = language_model
        self.vision_encoder = vision_encoder["model"] if isinstance(vision_encoder, dict) else vision_encoder
        self.vision_processor = vision_encoder["processor"] if isinstance(vision_encoder, dict) else None
        self.projection = projection_layer
        self.tokenizer = tokenizer
        self.config = config
        
        # Freeze components according to configuration (if they exist)
        if config.freeze_vision_encoder and self.vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
                
        if config.freeze_llm:
            for param in self.language_model.parameters():
                param.requires_grad = False
    
    @classmethod
    def from_config(cls, config: TrainingConfig) -> "MultimodalModel":
        """Create model from configuration."""
        logger.info("🏗️ Creating multimodal model...")
        
        # Import locally to avoid circular imports
        from .factory import ModelFactory
        
        # Load base language model
        language_model = ModelFactory.load_base_model(config)
        
        # Load tokenizer
        tokenizer = ModelFactory.load_tokenizer(config)
        
        # Load vision encoder (if any)
        vision_encoder = None
        vision_hidden_size = None  # Initialize for all cases
        
        if config.encoders and len(config.encoders) > 0:
            vision_config = list(config.encoders.values())[0]  # Take first encoder
            vision_encoder = ModelFactory.load_vision_encoder(vision_config.model)
        
        # Create projection layer (if vision encoder exists)
        projection = None
        if vision_encoder:
            # Get vision hidden size from returned dict
            vision_hidden_size = vision_encoder.get("hidden_size")
            if vision_hidden_size is None:
                # Fallback: get from model config
                vision_model = vision_encoder["model"]
                vision_config = vision_model.config
                
                if hasattr(vision_config, 'hidden_size'):
                    vision_hidden_size = vision_config.hidden_size
            elif hasattr(vision_config, 'vision_config') and hasattr(vision_config.vision_config, 'hidden_size'):
                vision_hidden_size = vision_config.vision_config.hidden_size
            elif hasattr(vision_config, 'projection_dim'):
                vision_hidden_size = vision_config.projection_dim
            else:
                # Fallback for CLIP and other models
                vision_hidden_size = 768  # Default for CLIP base
                logger.warning(f"⚠️ Could not determine vision hidden size, using default: {vision_hidden_size}")
        
        text_hidden_size = language_model.config.hidden_size
        
        # Create projection layer only if vision encoder exists
        projection_layer = None
        if vision_encoder and vision_hidden_size:
            projection_layer = ProjectionLayer(
                vision_hidden_size=vision_hidden_size,
                text_hidden_size=text_hidden_size,
                projection_type=config.projection.type
            )
        
        # Create model
        model = cls(
            language_model=language_model,
            vision_encoder=vision_encoder,
            projection_layer=projection_layer,
            tokenizer=tokenizer,
            config=config,
        )
        
        # Set up PEFT
        model.language_model = ModelFactory.setup_lora(model.language_model, config)
        
        # Freeze parameters (removed freeze_parameters method - handled in individual components)
        
        return model
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for multimodal model."""
        
        # Get embeddings from language model
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # If images exist, process them
        if images is not None:
            # Encode images
            vision_outputs = self.vision_encoder(images)
            vision_embeddings = vision_outputs.last_hidden_state
            
            # Project to text space
            projected_vision = self.projection(vision_embeddings)
            
            # TODO: Logic needed to insert vision embeddings at correct positions
            # This depends on how special tokens are marked in input_ids
            
        # Call language model
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        **kwargs
    ) -> torch.Tensor:
        """Text generation for multimodal model."""
        
        # Prepare embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        if images is not None:
            # Similar to forward pass
            vision_outputs = self.vision_encoder(images)
            vision_embeddings = vision_outputs.last_hidden_state
            projected_vision = self.projection(vision_embeddings)
            
            # TODO: Insert vision embeddings
        
        # Generation
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save model."""
        self.language_model.save_pretrained(save_directory)
        # TODO: Save projection layer and configuration 