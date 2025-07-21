"""
Configuration system for Any2Any Trainer.

Provides loading and validation of configurations from YAML files.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import yaml
from pydantic import BaseModel, validator
from transformers import TrainingArguments


class ModalityConfig(BaseModel):
    """Configuration for a single modality."""
    
    model: str
    freeze: bool = False
    tokenizer_type: str = "continuous"  # continuous, discrete
    max_length: Optional[int] = None
    processor_kwargs: Dict[str, Any] = {}


class ProjectionConfig(BaseModel):
    """Configuration for projection layers."""
    
    type: str = "mlp"  # mlp, linear, transformer
    hidden_size: int = 4096
    num_layers: int = 2
    dropout: float = 0.1


class LoRAConfig(BaseModel):
    """LoRA configuration."""
    
    r: int = 64
    alpha: int = 128
    dropout: float = 0.1
    target_modules: List[str] = []
    bias: str = "none"


def auto_detect_model_type(modalities: Dict[str, List[str]]) -> str:
    """
    Автоматически определяет model_type на основе modальностей.
    
    Логика:
    - {"input": ["text"], "output": ["text"]} -> 'standard' (обычная LLM)
    - одна модальность (не text) -> 'standard' (например, image | image)
    - мультимодальность с text output -> 'multimodal' 
    - сложные any2any случаи -> 'any2any'
    """
    input_modalities = set(modalities.get('input', []))
    output_modalities = set(modalities.get('output', []))
    
    # text -> text: standard LLM
    if input_modalities == {'text'} and output_modalities == {'text'}:
        return 'standard'
    
    # Одна немультимодальная модальность -> standard  
    if len(input_modalities) == 1 and len(output_modalities) == 1:
        input_mod = list(input_modalities)[0]
        output_mod = list(output_modalities)[0]
        if input_mod == output_mod and input_mod != 'text':
            return 'standard'  # например image -> image
    
    # Простые мультимодальные случаи -> multimodal
    # например: ["image", "text"] -> ["text"], ["image"] -> ["text"]
    if 'text' in output_modalities and len(output_modalities) == 1:
        return 'multimodal'
    
    # Сложные any2any случаи -> any2any
    return 'any2any'


class TrainingConfig(BaseModel):
    """Main configuration for training."""
    
    # Basic model parameters
    model_name_or_path: str
    model_type: Optional[str] = None  # автоопределение на основе modalities
    
    # Modality configuration
    modalities: Dict[str, List[str]] = {"input": ["text"], "output": ["text"]}
    encoders: Dict[str, ModalityConfig] = {}
    decoders: Dict[str, ModalityConfig] = {}
    
    # Vision encoder (for multimodal models)
    vision_encoder: Optional[str] = None
    projection_type: str = "mlp"  # mlp, linear, transformer
    
    # Projection layers
    projection: ProjectionConfig = ProjectionConfig()
    
    # Special tokens
    special_tokens: Dict[str, str] = {}
    
    # Dataset configuration
    dataset: List[str] = []
    conversation_field: str = "conversations"
    image_field: str = "image"
    max_seq_length: int = 2048
    
    # Training parameters (based on HF TrainingArguments)
    output_dir: str = "./output"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 0
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # LoRA and PEFT
    use_peft: bool = False
    lora: LoRAConfig = LoRAConfig()
    
    # Component freezing
    freeze_vision_encoder: bool = True
    freeze_audio_encoder: bool = True
    freeze_llm: bool = False
    train_projection_only: bool = False
    unfreeze_layers_patterns: List[str] = []
    
    # Additional parameters
    gradient_checkpointing: bool = False
    bf16: bool = True
    fp16: bool = False
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    
    # Evaluation settings
    eval_during_training: bool = True
    evaluation_strategy: str = "steps"  # no, steps, epoch
    
    # Logging
    report_to: str = "none"  # none, wandb, clearml, tensorboard
    run_name: Optional[str] = None
    
    # Generate examples during validation
    generate_eval_examples: bool = False
    max_new_tokens: int = 256
    
    def __init__(self, **data):
        super().__init__(**data)
        self._auto_detect_model_type()
    
    def _auto_detect_model_type(self):
        """Автоматически определяет model_type на основе modalities."""
        if not self.model_type:
            self.model_type = auto_detect_model_type(self.modalities)
            print(f"🤖 Автоопределен model_type '{self.model_type}' на основе modalities: {self.modalities}")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v is None:
            return v  # Будет автоопределен в __init__
        valid_types = ['standard', 'multimodal', 'any2any', 'unified']
        if v not in valid_types:
            raise ValueError(f"model_type must be one of: {valid_types}, got {v}")
        return v

    @validator('encoders', pre=True)
    def validate_encoders(cls, v):
        if isinstance(v, dict):
            return {k: ModalityConfig(**val) if isinstance(val, dict) else val 
                    for k, val in v.items()}
        return v

    @validator('decoders', pre=True) 
    def validate_decoders(cls, v):
        if isinstance(v, dict):
            return {k: ModalityConfig(**val) if isinstance(val, dict) else val 
                    for k, val in v.items()}
        return v


class ConfigManager:
    """Configuration manager."""
    
    @staticmethod
    def from_dict(config_dict: dict) -> TrainingConfig:
        """Create configuration from dictionary."""
        return TrainingConfig(**config_dict)
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> TrainingConfig:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return TrainingConfig(**config_dict)
    
    @staticmethod
    def save_config(config: TrainingConfig, save_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config.dict(), f, default_flow_style=False, 
                     allow_unicode=True, sort_keys=False)
    
    @staticmethod
    def to_training_arguments(config: TrainingConfig) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        training_args_dict = {
            'output_dir': config.output_dir,
            'per_device_train_batch_size': config.per_device_train_batch_size,
            'per_device_eval_batch_size': config.per_device_eval_batch_size,
            'gradient_accumulation_steps': config.gradient_accumulation_steps,
            'num_train_epochs': config.num_train_epochs,
            'learning_rate': config.learning_rate,
            'warmup_steps': config.warmup_steps,
            'logging_steps': config.logging_steps,
            'save_steps': config.save_steps,
            'eval_steps': config.eval_steps,
            'save_total_limit': config.save_total_limit,
            'gradient_checkpointing': config.gradient_checkpointing,
            'bf16': config.bf16,
            'fp16': config.fp16,
            'dataloader_num_workers': config.dataloader_num_workers,
            'remove_unused_columns': config.remove_unused_columns,
            'report_to': config.report_to,
        }
        
        if config.run_name:
            training_args_dict['run_name'] = config.run_name
            
        return TrainingArguments(**training_args_dict)
    
    @staticmethod
    def validate_config(config: TrainingConfig) -> None:
        """Validate configuration."""
        # Check modalities
        input_modalities = set(config.modalities.get("input", []))
        output_modalities = set(config.modalities.get("output", []))
        
        # Check encoder availability for input modalities
        for modality in input_modalities:
            if modality != "text" and modality not in config.encoders:
                raise ValueError(f"Encoder not found for modality: {modality}")
        
        # Check decoder availability for output modalities (except text)
        for modality in output_modalities:
            if modality != "text" and modality not in config.decoders:
                raise ValueError(f"Decoder not found for modality: {modality}")
        
        # Check LoRA configuration
        if config.use_peft and not config.lora.target_modules:
            raise ValueError("target_modules must be specified when using LoRA")
        
        print("✅ Configuration passed validation") 