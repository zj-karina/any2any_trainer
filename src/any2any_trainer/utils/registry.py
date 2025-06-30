"""
Model and tokenizer registration system.

Allows easy addition of new encoders, decoders and tokenizers for various modalities.
"""

from typing import Dict, Callable, Any, Type, Optional
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer


class ModalityEncoder(ABC):
    """Base class for modality encoders."""
    
    @abstractmethod
    def encode(self, inputs: Any) -> torch.Tensor:
        """Encode input data into embeddings."""
        pass
    
    @abstractmethod 
    def get_output_dim(self) -> int:
        """Get output embeddings dimension."""
        pass


class ModalityDecoder(ABC):
    """Base class for modality decoders."""
    
    @abstractmethod
    def decode(self, embeddings: torch.Tensor) -> Any:
        """Decode embeddings into output data."""
        pass
    
    @abstractmethod
    def get_input_dim(self) -> int:
        """Get input embeddings dimension."""
        pass


class ModalityTokenizer(ABC):
    """Base class for modality tokenizers."""
    
    @abstractmethod
    def tokenize(self, inputs: Any) -> Dict[str, torch.Tensor]:
        """Tokenize input data."""
        pass
    
    @abstractmethod
    def detokenize(self, tokens: torch.Tensor) -> Any:
        """Detokenize tokens back to data."""
        pass


class ModelRegistry:
    """Model registry for various modalities."""
    
    _encoders: Dict[str, Dict[str, Callable]] = {}
    _decoders: Dict[str, Dict[str, Callable]] = {}
    
    @classmethod
    def register_encoder(cls, modality: str, model_name: str, encoder_class: Callable):
        """Register encoder for modality."""
        if modality not in cls._encoders:
            cls._encoders[modality] = {}
        cls._encoders[modality][model_name] = encoder_class
    
    @classmethod
    def register_decoder(cls, modality: str, model_name: str, decoder_class: Callable):
        """Register decoder for modality."""
        if modality not in cls._decoders:
            cls._decoders[modality] = {}
        cls._decoders[modality][model_name] = decoder_class
    
    @classmethod
    def get_encoder(cls, modality: str, model_name: str) -> Callable:
        """Get encoder for modality."""
        if modality not in cls._encoders:
            raise ValueError(f"Modality {modality} is not registered")
        if model_name not in cls._encoders[modality]:
            raise ValueError(f"Model {model_name} for modality {modality} not found")
        return cls._encoders[modality][model_name]
    
    @classmethod
    def get_decoder(cls, modality: str, model_name: str) -> Callable:
        """Get decoder for modality."""
        if modality not in cls._decoders:
            raise ValueError(f"Modality {modality} is not registered")
        if model_name not in cls._decoders[modality]:
            raise ValueError(f"Model {model_name} for modality {modality} not found")
        return cls._decoders[modality][model_name]
    
    @classmethod
    def list_encoders(cls, modality: Optional[str] = None) -> Dict[str, list]:
        """Get list of available encoders."""
        if modality:
            return {modality: list(cls._encoders.get(modality, {}).keys())}
        return {mod: list(models.keys()) for mod, models in cls._encoders.items()}
    
    @classmethod
    def list_decoders(cls, modality: Optional[str] = None) -> Dict[str, list]:
        """Get list of available decoders."""
        if modality:
            return {modality: list(cls._decoders.get(modality, {}).keys())}
        return {mod: list(models.keys()) for mod, models in cls._decoders.items()}


class TokenizerRegistry:
    """Tokenizer registry for various modalities."""
    
    _tokenizers: Dict[str, Dict[str, Callable]] = {}
    
    @classmethod
    def register_tokenizer(cls, modality: str, tokenizer_name: str, tokenizer_class: Callable):
        """Register tokenizer for modality."""
        if modality not in cls._tokenizers:
            cls._tokenizers[modality] = {}
        cls._tokenizers[modality][tokenizer_name] = tokenizer_class
    
    @classmethod
    def get_tokenizer(cls, modality: str, tokenizer_name: str) -> Callable:
        """Get tokenizer for modality."""
        if modality not in cls._tokenizers:
            raise ValueError(f"Modality {modality} is not registered")
        if tokenizer_name not in cls._tokenizers[modality]:
            raise ValueError(f"Tokenizer {tokenizer_name} for modality {modality} not found")
        return cls._tokenizers[modality][tokenizer_name]
    
    @classmethod
    def list_tokenizers(cls, modality: Optional[str] = None) -> Dict[str, list]:
        """Get list of available tokenizers."""
        if modality:
            return {modality: list(cls._tokenizers.get(modality, {}).keys())}
        return {mod: list(tokenizers.keys()) for mod, tokenizers in cls._tokenizers.items()}


# Basic encoders for popular modalities

class CLIPVisionEncoder(ModalityEncoder):
    """CLIP encoder for images."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = AutoModel.from_pretrained(model_name).vision_model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size
    
    def encode(self, images) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state  # [batch, seq_len, hidden_size]
    
    def get_output_dim(self) -> int:
        return self.output_dim


class WhisperEncoder(ModalityEncoder):
    """Whisper encoder for audio."""
    
    def __init__(self, model_name: str = "openai/whisper-base"):
        self.model = AutoModel.from_pretrained(model_name).encoder
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.output_dim = self.model.config.d_model
    
    def encode(self, audio) -> torch.Tensor:
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16000)
        outputs = self.model(inputs.input_features)
        return outputs.last_hidden_state
    
    def get_output_dim(self) -> int:
        return self.output_dim


class TextEncoder(ModalityEncoder):
    """Text encoder based on transformer."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size
    
    def encode(self, texts) -> torch.Tensor:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state
    
    def get_output_dim(self) -> int:
        return self.output_dim


# Register basic encoders
ModelRegistry.register_encoder("image", "clip", CLIPVisionEncoder)
ModelRegistry.register_encoder("audio", "whisper", WhisperEncoder)  
ModelRegistry.register_encoder("text", "transformer", TextEncoder)

# Basic tokenizers

class DiscreteTokenizer(ModalityTokenizer):
    """Discrete tokenizer for modalities."""
    
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
    
    def tokenize(self, inputs: Any) -> Dict[str, torch.Tensor]:
        # Placeholder - in reality this would be VQ-VAE or similar method
        return {"input_ids": torch.randint(0, self.vocab_size, (1, 100))}
    
    def detokenize(self, tokens: torch.Tensor) -> Any:
        # Placeholder for detokenization
        return tokens


class ContinuousTokenizer(ModalityTokenizer):
    """Continuous tokenizer for modalities."""
    
    def __init__(self, embed_dim: int = 768):
        self.embed_dim = embed_dim
    
    def tokenize(self, inputs: Any) -> Dict[str, torch.Tensor]:
        # Direct embedding passthrough
        if isinstance(inputs, torch.Tensor):
            return {"embeddings": inputs}
        return {"embeddings": torch.randn(1, 100, self.embed_dim)}
    
    def detokenize(self, tokens: torch.Tensor) -> Any:
        return tokens


# Register basic tokenizers
TokenizerRegistry.register_tokenizer("image", "discrete", DiscreteTokenizer)
TokenizerRegistry.register_tokenizer("audio", "discrete", DiscreteTokenizer)
TokenizerRegistry.register_tokenizer("image", "continuous", ContinuousTokenizer)
TokenizerRegistry.register_tokenizer("audio", "continuous", ContinuousTokenizer)
TokenizerRegistry.register_tokenizer("text", "continuous", ContinuousTokenizer) 