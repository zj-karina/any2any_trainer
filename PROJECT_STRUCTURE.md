# 🏗️ Any2Any Trainer - Architecture and Project Structure

This document describes the architecture, design principles, and structure of the Any2Any Trainer library.

## 🎯 Design Philosophy

### Core Principles

1. **Simplicity First**: Simple things should be simple, complex things should be possible
2. **HuggingFace Native**: Direct usage of HF models without complex wrappers
3. **Modularity**: Each component is independent and can be extended
4. **Configuration-Driven**: Everything controlled through YAML configs
5. **Performance**: Support for modern training techniques (FSDP, DeepSpeed, Flash Attention)

### Architecture Goals

- ✅ **Plug & Play**: Use any HF model immediately
- ✅ **Scalability**: From single GPU to multi-node training
- ✅ **Extensibility**: Easy to add new modalities and architectures
- ✅ **Maintainability**: Clean, well-documented code
- ✅ **Community**: Built on proven libraries and patterns

## 📁 Project Structure

```
any2any_trainer/
├── 📄 README.md                    # Main documentation
├── 📄 QUICKSTART.md                # Quick start guide
├── 📄 HF_MODELS_USAGE.md           # HuggingFace models usage guide
├── 📄 PROJECT_STRUCTURE.md         # This file
├── 📄 LICENSE                      # Apache 2.0 license
├── 📄 pyproject.toml               # Poetry dependencies and settings
├── 📄 .gitignore                   # Git ignore rules
│
├── 📂 src/                         # Source code
│   └── 📂 any2any_trainer/
│       ├── 📄 __init__.py
│       │
│       ├── 📂 models/              # Model architectures
│       │   ├── 📄 __init__.py
│       │   ├── 📄 factory.py       # Model loading and creation
│       │   ├── 📄 multimodal.py    # Multimodal models (LLaVA-style)
│       │   ├── 📄 any2any.py       # Any-to-any models (AnyGPT-style)
│       │   │
│       │   ├── 📂 encoders/        # Modality encoders
│       │   │   ├── 📄 __init__.py
│       │   │   ├── 📄 text.py      # Text encoders
│       │   │   ├── 📄 vision.py    # Vision encoders (CLIP, etc.)
│       │   │   ├── 📄 audio.py     # Audio encoders (Whisper, etc.)
│       │   │   └── 📄 video.py     # Video encoders
│       │   │
│       │   ├── 📂 decoders/        # Modality decoders
│       │   │   ├── 📄 __init__.py
│       │   │   ├── 📄 text.py      # Text generation
│       │   │   ├── 📄 image.py     # Image generation
│       │   │   └── 📄 audio.py     # Audio generation
│       │   │
│       │   └── 📂 projections/     # Projection layers
│       │       ├── 📄 __init__.py
│       │       ├── 📄 linear.py    # Linear projections
│       │       ├── 📄 mlp.py       # MLP projections
│       │       └── 📄 attention.py # Attention-based projections
│       │
│       ├── 📂 data/                # Data processing
│       │   ├── 📄 __init__.py
│       │   ├── 📄 collator.py      # Data collation for batching
│       │   │
│       │   ├── 📂 datasets/        # Dataset implementations
│       │   │   ├── 📄 __init__.py
│       │   │   ├── 📄 conversation.py    # Conversational datasets
│       │   │   ├── 📄 multimodal.py     # Multimodal datasets
│       │   │   └── 📄 any2any.py        # Any-to-any datasets
│       │   │
│       │   ├── 📂 processors/      # Data processors
│       │   │   ├── 📄 __init__.py
│       │   │   ├── 📄 text.py      # Text processing
│       │   │   ├── 📄 image.py     # Image processing
│       │   │   ├── 📄 audio.py     # Audio processing
│       │   │   └── 📄 video.py     # Video processing
│       │   │
│       │   └── 📂 tokenizers/      # Modality tokenizers
│       │       ├── 📄 __init__.py
│       │       ├── 📄 discrete.py  # Discrete tokenization
│       │       └── 📄 continuous.py # Continuous tokenization
│       │
│       ├── 📂 training/            # Training components
│       │   ├── 📄 __init__.py
│       │   │
│       │   ├── 📂 trainers/        # Custom trainers
│       │   │   ├── 📄 __init__.py
│       │   │   ├── 📄 multimodal.py     # Multimodal SFT trainer
│       │   │   ├── 📄 any2any.py        # Any-to-any trainer
│       │   │   └── 📄 base.py           # Base trainer class
│       │   │
│       │   ├── 📂 losses/          # Loss functions
│       │   │   ├── 📄 __init__.py
│       │   │   ├── 📄 cross_entropy.py # Standard CE loss
│       │   │   ├── 📄 multimodal.py    # Multimodal losses
│       │   │   └── 📄 contrastive.py   # Contrastive losses
│       │   │
│       │   └── 📂 callbacks/       # Training callbacks
│       │       ├── 📄 __init__.py
│       │       ├── 📄 logging.py   # Logging callbacks
│       │       └── 📄 evaluation.py # Evaluation callbacks
│       │
│       └── 📂 utils/               # Utilities
│           ├── 📄 __init__.py
│           ├── 📄 config.py        # Configuration system
│           ├── 📄 registry.py      # Model and component registry
│           ├── 📄 logging.py       # Logging utilities
│           └── 📄 io.py            # Input/output utilities
│
├── 📂 scripts/                     # Training scripts
│   ├── 📄 train_multimodal.py      # Main multimodal training script
│   └── 📄 train_any2any.py         # Any-to-any training script
│
├── 📂 configs/                     # Configuration files
│   ├── 📂 sft/                     # Supervised fine-tuning configs
│   │   ├── 📄 simple_hf_training.yaml      # Simple HF model training
│   │   ├── 📄 hf_multimodal_training.yaml  # HF multimodal training
│   │   └── 📄 llava_style_training.yaml    # LLaVA-style training
│   │
│   └── 📂 any2any/                 # Any-to-any configs
│       └── 📄 anygpt_style_training.yaml   # AnyGPT-style training
│
├── 📂 accelerate/                  # Accelerate configurations
│   └── 📄 fsdp_config.yaml         # FSDP configuration
│
└── 📂 deepspeed_configs/           # DeepSpeed configurations
    └── 📄 zero2_config.json        # ZeRO Stage 2 config
```

## 🧩 Core Components

### 1. Model Factory (`src/models/factory.py`)

**Purpose**: Centralized model loading and creation

```python
def load_base_model(config: TrainingConfig) -> nn.Module:
    """Load any HuggingFace model directly."""
    return AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=get_torch_dtype(config),
        device_map="auto" if config.use_device_map else None,
        trust_remote_code=config.trust_remote_code,
    )

def setup_peft(model: nn.Module, config: TrainingConfig) -> nn.Module:
    """Automatically configure LoRA/QLoRA."""
    if not config.use_peft:
        return model
    # LoRA configuration...
```

**Key Features**:
- Direct HuggingFace model loading
- Automatic PEFT/LoRA setup
- Memory optimization (quantization, device mapping)
- Error handling and validation

### 2. Configuration System (`src/utils/config.py`)

**Purpose**: Type-safe configuration management using Pydantic

```python
@dataclass
class TrainingConfig:
    # Model configuration
    model_name_or_path: str
    model_type: str = "auto"
    trust_remote_code: bool = False
    
    # Training parameters (inherits from HF TrainingArguments)
    per_device_train_batch_size: int = 1
    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    
    # PEFT configuration
    use_peft: bool = False
    lora: Optional[LoRAConfig] = None
    
    # Modality configuration
    modalities: Optional[ModalityConfig] = None
```

**Key Features**:
- Type validation with Pydantic
- Automatic defaults
- Environment variable support
- Hierarchical configuration loading

### 3. Multimodal Models (`src/models/multimodal.py`)

**Purpose**: LLaVA-style multimodal architectures

```python
class MultimodalModel(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        # Load base LLM
        self.llm = load_base_model(config)
        
        # Load vision encoder
        if config.vision_encoder:
            self.vision_encoder = load_vision_encoder(config.vision_encoder)
            self.vision_projection = create_projection(
                input_dim=self.vision_encoder.config.hidden_size,
                output_dim=self.llm.config.hidden_size,
                projection_type=config.projection_type
            )
    
    def forward(self, input_ids, images=None, **kwargs):
        # Encode images if present
        if images is not None:
            image_features = self.vision_encoder(images)
            image_embeds = self.vision_projection(image_features)
            # Merge with text embeddings...
        
        # Forward through LLM
        return self.llm(input_ids=input_ids, **kwargs)
```

### 4. Data Processing (`src/data/`)

**Purpose**: Efficient data loading and processing

- **Datasets**: Handle different data formats (conversations, instructions, multimodal)
- **Collators**: Batch data efficiently with proper padding
- **Processors**: Convert raw data to model inputs

### 5. Training Components (`src/training/`)

**Purpose**: Custom training logic and optimization

- **Trainers**: Extend HuggingFace Trainer for multimodal scenarios
- **Losses**: Specialized loss functions for different modalities
- **Callbacks**: Custom training callbacks for logging and evaluation

## 🔄 Data Flow

### 1. Configuration Loading
```
YAML Config → Pydantic Validation → TrainingConfig Object
```

### 2. Model Creation
```
Config → ModelFactory → Base Model → PEFT Setup → Final Model
```

### 3. Data Processing
```
Raw Data → Dataset → Processor → Collator → Batched Data
```

### 4. Training Loop
```
Batched Data → Model Forward → Loss Computation → Backward → Optimization
```

## 🎨 Extension Points

### Adding New Modalities

1. **Create Encoder** (`src/models/encoders/new_modality.py`):
```python
class NewModalityEncoder(nn.Module):
    def __init__(self, model_name: str):
        # Load pretrained encoder
        pass
    
    def encode(self, inputs) -> torch.Tensor:
        # Encode inputs to embeddings
        pass
```

2. **Register Encoder** (`src/utils/registry.py`):
```python
ENCODER_REGISTRY.register("new_modality", NewModalityEncoder)
```

3. **Create Processor** (`src/data/processors/new_modality.py`):
```python
class NewModalityProcessor:
    def process(self, data) -> Dict[str, Any]:
        # Process raw data to model inputs
        pass
```

4. **Update Configuration**:
```python
@dataclass
class ModalityConfig:
    new_modality_encoder: Optional[str] = None
```

### Adding New Architectures

1. **Create Model Class** (`src/models/new_architecture.py`):
```python
class NewArchitecture(nn.Module):
    def __init__(self, config: TrainingConfig):
        # Initialize architecture
        pass
```

2. **Register in Factory**:
```python
MODEL_REGISTRY.register("new_architecture", NewArchitecture)
```

3. **Create Trainer** (`src/training/trainers/new_architecture.py`):
```python
class NewArchitectureTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Custom loss computation
        pass
```

## ⚙️ Configuration System

### Hierarchical Loading

1. **Default Values**: Hardcoded in dataclass definitions
2. **Config File**: YAML file specified by user
3. **Environment Variables**: Override specific values
4. **Command Line**: Override specific values (future feature)

### Example Configuration Flow

```yaml
# config.yaml
model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
use_peft: true
lora:
  r: 64
  alpha: 128
```

```python
# Loading process
config = load_config("config.yaml")
# 1. Load defaults from TrainingConfig
# 2. Override with config.yaml values
# 3. Override with environment variables
# 4. Validate with Pydantic
```

## 🚀 Performance Optimizations

### Memory Efficiency

1. **Gradient Checkpointing**: Reduce memory usage during backpropagation
2. **Mixed Precision**: FP16/BF16 training for faster computation
3. **LoRA/QLoRA**: Parameter-efficient fine-tuning
4. **DeepSpeed ZeRO**: Distribute optimizer states and parameters

### Computation Efficiency

1. **Flash Attention**: Optimized attention computation
2. **Torch Compile**: JIT compilation for faster execution
3. **Data Loading**: Multi-worker data loading with proper pinning
4. **FSDP**: Fully Sharded Data Parallel for large models

### Distributed Training

1. **Accelerate Integration**: Seamless multi-GPU training
2. **FSDP Support**: For models that don't fit on single GPU
3. **DeepSpeed Integration**: Advanced optimization strategies
4. **Gradient Accumulation**: Simulate larger batch sizes

## 🔍 Monitoring and Debugging

### Logging System

- **Rich Console**: Beautiful terminal output
- **File Logging**: Persistent logs for analysis
- **Structured Logging**: JSON format for parsing
- **Performance Metrics**: Training speed and memory usage

### Integration with Tracking

- **Weights & Biases**: Experiment tracking and visualization
- **TensorBoard**: Local experiment tracking
- **ClearML**: Enterprise experiment tracking
- **Custom Callbacks**: Easy integration with other tools

## 🧪 Testing Strategy

### Unit Tests
- Model component testing
- Configuration validation
- Data processing verification

### Integration Tests
- End-to-end training workflows
- Multi-GPU training validation
- Different model combinations

### Performance Tests
- Memory usage benchmarks
- Training speed measurements
- Scalability testing

## 📈 Future Enhancements

### Planned Features

1. **Video Modality**: Complete video processing pipeline
2. **3D Models**: Support for 3D data (point clouds, meshes)
3. **API Integration**: Support for external APIs (OpenAI, Anthropic)
4. **Auto-tuning**: Automatic hyperparameter optimization
5. **Model Compression**: Quantization and pruning support

### Architecture Improvements

1. **Plugin System**: Dynamic loading of extensions
2. **Registry Enhancement**: More flexible component registration
3. **Configuration UI**: Web interface for configuration creation
4. **Distributed Data**: Support for distributed datasets

## 🤝 Contributing Guidelines

### Code Organization

- **Single Responsibility**: Each module has one clear purpose
- **Dependency Injection**: Components receive dependencies explicitly
- **Interface Segregation**: Small, focused interfaces
- **Documentation**: Comprehensive docstrings and examples

### Development Workflow

1. **Fork & Clone**: Standard GitHub workflow
2. **Feature Branches**: One feature per branch
3. **Tests**: Add tests for new features
4. **Documentation**: Update docs for user-facing changes
5. **Pull Request**: Comprehensive description and testing

### Code Style

- **Type Hints**: All public functions have type annotations
- **Docstrings**: Google-style docstrings
- **Formatting**: Black + isort for consistency
- **Linting**: flake8 + mypy for quality

---

This architecture ensures that Any2Any Trainer remains **simple to use** for basic scenarios while being **powerful and extensible** for advanced use cases. The modular design allows users to pick and choose components they need, while the configuration system provides a consistent interface across all features. 