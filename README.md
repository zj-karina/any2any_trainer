# 🚀 Any2Any Trainer - Universal Multimodal Training Toolkit

> **New here?** Jump to [Installation](#-installation) → [Quick Start](#-quick-start) → [Examples](#-example-yaml-configurations)

## Table of Contents
- [✨ Features](#-description-and-features)
- [📦 Installation](#-installation) 
- [🚀 Quick Start](#-quick-start)
- [📝 Configuration Examples](#-example-yaml-configurations)
- [🧠 Smart Configuration](#-smart-configuration)
- [📚 Documentation](#-documentation)
- [🎯 Roadmap](#-development-roadmap)

## ✨ Description and Features

This is a **universal**, **customizable**, **user-friendly**, and **efficient** toolkit for training any-to-any multimodal models. 

Simply define a YAML configuration with HF TrainingArguments parameters and specific parameters for each modality.

### 🛠️ Toolkit Foundation

**Core Libraries:**
- **Core:** PyTorch, Transformers, TRL
- **Distributed Training:** Accelerate, FSDP, DeepSpeed (Zero 2/3)
- **Acceleration:** vLLM, Flash Attention, SDPA
- **Build and Installation:** Poetry
- **Result Logging:** wandb, clearml, tensorboard

## 📚 Supported Modalities and Methods

### Modalities
- **Text:** Standard LLMs (Llama, Qwen, Mistral, etc.)
- **Images:** Vision encoders (CLIP, SigLip, ViT, etc.)
- **Audio:** Speech encoders (Whisper, WavLM, Wav2Vec2, etc.)
- **Video:** Video encoders (VideoMAE, I3D, etc.)
- **Custom:** Easily extensible architecture

### Training Methods
- **SFT (Supervised Fine-Tuning):** Standard instruction fine-tuning
- **Multimodal SFT:** Training on multimodal dialogues
- **LoRA/QLoRA:** Efficient fine-tuning with low-rank adapters
- **Full Fine-tuning:** Complete parameter fine-tuning
- **Projection Training:** Training only projection layers

### Architectures
- **Encoder-Decoder:** Separate encoders for each modality + shared decoder
- **Unified:** Single model for all modalities (like AnyGPT)
- **Projection-based:** LLM + projection adapters (like LLaVA)

## 🧠 Smart Configuration

### 🚀 NEW: Automatic Model Type Detection

Больше **НЕ НУЖНО** указывать `model_type` в конфигурации! Система автоматически определит тип модели на основе указанных модальностей:

```yaml
# 🤖 Автоматически определится как "standard"
modalities:
  input: ["text"]
  output: ["text"]

# 🤖 Автоматически определится как "multimodal"  
modalities:
  input: ["image", "text"]
  output: ["text"]

# 🤖 Автоматически определится как "any2any"
modalities:
  input: ["text", "image", "audio"]
  output: ["text", "image", "audio"]
```

### Логика автоопределения:
- **`standard`**: `text → text` или одинаковые модальности (например, `image → image`)
- **`multimodal`**: Несколько входных модальностей с выходом в текст
- **`any2any`**: Сложные комбинации множественных модальностей

## 🚀 How to Use

### 📦 Installation

**Quick Setup:**
```bash
cd any2any_trainer
poetry install

## 🚀 Quick Start

### ✅ Test Installation (VERIFIED WORKING)
```bash
poetry run python test_installation.py
```

### ✅ Рабочие конфигурации (ПРОВЕРЕННЫЕ)

| Конфигурация | Модель | Тип | Статус |
|--------------|--------|-----|--------|
| `minimal_working.yaml` | GPT-2 | standard | ✅ Работает |
| `simple_hf_training.yaml` | Qwen-7B | standard | ✅ Работает |
| `llava_training.yaml` | DialoGPT + CLIP | multimodal | ✅ Работает |

```bash
# Быстрый тест (самый простой)
poetry run python scripts/train_multimodal.py configs/sft/minimal_working.yaml

# Основной тест функциональности
poetry run python test_installation.py  # Confirms: Model loading, LoRA, CUDA, forward pass
```

### Training Configuration Example
```bash
# Create simple config - model_type автоопределится как "standard"!
echo 'model_name_or_path: "gpt2"
modalities:
  input: ["text"]
  output: ["text"]
dataset: ["wikitext", "wikitext-2-raw-v1"]
output_dir: "./my_first_model"
per_device_train_batch_size: 1
num_train_epochs: 1
use_peft: true
lora:
  r: 8
  alpha: 16
  target_modules: ["c_attn"]' > quick_test.yaml
```

### 🏃‍♂️ Example Training Commands

```bash



# Простая HuggingFace модель (Qwen + LoRA)  
poetry run python scripts/train_multimodal.py configs/sft/simple_hf_training.yaml

# Мультимодальная модель (LLaVA-style)
poetry run python scripts/train_multimodal.py configs/sft/llava_training.yaml
```

### 🤗 Simple HuggingFace Model Usage

#### Minimal Configuration

```yaml
# Just specify any model from HF Hub! model_type автоопределится как "standard"
model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
modalities:
  input: ["text"]
  output: ["text"]
dataset: ["tatsu-lab/alpaca"]
use_peft: true
lora:
  target_modules: ["q_proj", "v_proj"]
per_device_train_batch_size: 2
output_dir: "./my_model"
```

**Run:**
```bash
python scripts/train_multimodal.py my_config.yaml
```

### 📝 Example YAML Configurations

#### LLaVA-style Model Training (image → text)

```yaml
# configs/sft/llava_style_training.yaml
model_name_or_path: "microsoft/DialoGPT-medium"
# model_type автоматически определится как "multimodal"!

# IMPORTANT: Convert LLaVA dataset first!
# Use: python scripts/convert_llava_to_conversations.py llava_data.jsonl converted_data.jsonl
dataset:
  - "converted_llava_data.jsonl"  # Path to converted LLaVA dataset
  
modalities:
  input: ["image", "text"]
  output: ["text"]

encoders:
  image:
    model: "openai/clip-vit-base-patch32"
    freeze: true

projection:
  type: "mlp"  # mlp, linear, transformer

per_device_train_batch_size: 2
per_device_eval_batch_size: 2
num_train_epochs: 3
learning_rate: 2e-5
gradient_accumulation_steps: 4
max_seq_length: 2048

# Specific parameters for multimodal training
freeze_vision_encoder: true
freeze_llm: false
train_projection_only: false

use_peft: true
lora_r: 64
lora_alpha: 128
lora_target_modules:
  - "q_proj"
  - "v_proj"
  - "k_proj"
  - "o_proj"

conversation_field: "conversations"
image_field: "image"
```

#### Any-to-Any Model Training (any modality → any modality)

```yaml
# configs/any2any/anygpt_style_training.yaml
model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
# model_type автоматически определится как "any2any"!

modalities:
  input: ["text", "image", "audio", "video"]
  output: ["text", "image", "audio"]

# Encoders for each modality
encoders:
  image:
    model: "openai/clip-vit-large-patch14"
    freeze: true
    tokenizer_type: "discrete"
  audio:
    model: "openai/whisper-base"
    freeze: true
    tokenizer_type: "discrete"
  video:
    model: "microsoft/videoMAE-base"
    freeze: true
    tokenizer_type: "discrete"

# Decoders for output modalities
decoders:
  image:
    model: "stabilityai/stable-diffusion-2-1"
    freeze: false
  audio:
    model: "microsoft/speecht5_tts"
    freeze: false

# Special tokens for modalities
special_tokens:
  image_start: "<img>"
  image_end: "</img>"
  audio_start: "<aud>"
  audio_end: "</aud>"
  video_start: "<vid>"
  video_end: "</vid>"

dataset:
  - "custom/multimodal_conversations"

per_device_train_batch_size: 1
gradient_accumulation_steps: 16
num_train_epochs: 2
learning_rate: 1e-5
max_seq_length: 4096

use_peft: true
lora_r: 128
```

## 🏗️ Library Architecture

```
any2any_trainer/
├── src/
│   ├── models/           # Model architectures
│   │   ├── encoders/     # Encoders for different modalities
│   │   ├── decoders/     # Decoders for different modalities
│   │   ├── projectors/   # Projection layers
│   │   └── unified/      # Unified models (AnyGPT-style)
│   ├── data/            # Data processing
│   │   ├── datasets/    # Datasets for different modalities
│   │   ├── collators/   # Data collators
│   │   └── tokenizers/  # Modality tokenizers
│   ├── training/        # Training
│   │   ├── trainers/    # Custom trainers
│   │   ├── losses/      # Loss functions
│   │   └── callbacks/   # Training callbacks
│   └── utils/           # Utilities
├── scripts/             # Launch scripts
├── configs/             # YAML configurations
├── accelerate/          # Accelerate configs
└── deepspeed_configs/   # DeepSpeed configs
```

## 📊 Data Format

**📋 [READ DATA_FORMAT.md](DATA_FORMAT.md) - Complete specification**

Any2Any Trainer uses **single standard format** based on OpenAI-style conversations:

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "What's in the image?",
      "image": "path/to/image.jpg"  // optional multimodal data
    },
    {
      "role": "assistant", 
      "content": "The image shows a cat sitting on a window."
    }
  ]
}
```

**⚠️ Important**: We do NOT provide automatic format conversion. Convert your data to this standard format before training.

**✅ Compatible with**: OpenAI API, HuggingFace chat templates, TRL, effective_llm_alignment

**🔧 Format Converters Available:**
- **LLaVA Dataset**: 
  - `scripts/convert_llava_to_conversations.py` - converts LLaVA format to conversations format
  - `scripts/download_and_convert_llava.py` - downloads and converts LLaVA-Instruct-150K automatically

**📥 Quick LLaVA Setup:**
```bash
# Option 1: Automatic download and conversion
poetry run python scripts/download_and_convert_llava.py

# Option 2: Manual conversion (if you have LLaVA data)
poetry run python scripts/convert_llava_to_conversations.py input.jsonl output.jsonl

# Then train with:
python scripts/train_multimodal.py configs/sft/llava_training.yaml
```

### ❌ Model Type Errors
Убедитесь что `model_type` в YAML один из:
- `"standard"` - обычные HuggingFace модели
- `"multimodal"` - мультимодальные модели (LLaVA-style) 
- `"any2any"` - any-to-any модели

## 📚 Documentation

### 📖 Detailed Guides
- 🤗 [**HF_MODELS_USAGE.md**](HF_MODELS_USAGE.md) - Complete guide for using HuggingFace models

### 📋 Quick Reference
- [Installation](#-installation)
- [Quick Start](#-quick-start) 
- [Configuration Examples](#-example-yaml-configurations)
- [Supported Models](#-supported-modalities-and-methods)

## 🎯 Development Roadmap

- [x] Direct HuggingFace model usage
- [x] Configuration system and modular architecture
- [ ] Complete implementation of trainers and datasets
- [ ] Video modality support
- [ ] Integration with Gemini, GPT-4V API
- [ ] 3D model support
- [ ] Web interface for configuration creation
- [ ] Automatic hyperparameter tuning

## 🤝 How to Contribute

We welcome contributions to the library development! Main areas:
- Adding new encoders/decoders
- Supporting new modalities
- Performance optimization
- Documentation improvements

## 📄 License

Apache-2.0 License 