# 🚀 Any2Any Trainer - Quick Start Guide

This is a step-by-step guide to quickly get started with the toolkit.

## 📦 Installation

```bash
# 1. Clone the repository (if you haven't already)
cd any2any_trainer

# 2. Install Poetry (if you haven't already)
pip install poetry

# 3. Install dependencies
poetry install

# 4. Set up environment (optional)
export HF_HOME=/path/to/your/hf/cache

# 5. Login to services (optional)
poetry run huggingface-cli login
poetry run wandb login
```

## 🏃‍♂️ Simplest Examples

### 1. Training Any HuggingFace Model (Minimum Config)

**Create config file `quick_start.yaml`:**

```yaml
# Basic configuration - just the essentials!
model_name_or_path: "microsoft/DialoGPT-small"
dataset: ["tatsu-lab/alpaca"]
per_device_train_batch_size: 2
num_train_epochs: 1
output_dir: "./my_first_model"
```

**Run:**
```bash
PYTHONPATH="${PYTHONPATH}:src/" poetry run python \
    scripts/train_multimodal.py \
    quick_start.yaml
```

### 2. Training with LoRA (Efficient Training)

**Config `lora_training.yaml`:**
```yaml
model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
dataset: ["tatsu-lab/alpaca"]

# LoRA configuration
use_peft: true
lora:
  r: 32
  alpha: 64
  target_modules: ["q_proj", "v_proj"]
  dropout: 0.1

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
num_train_epochs: 3
learning_rate: 1e-4
output_dir: "./lora_model"

# Save configuration
save_strategy: "epoch"
logging_steps: 50
```

**Run:**
```bash
PYTHONPATH="${PYTHONPATH}:src/" poetry run python \
    scripts/train_multimodal.py \
    lora_training.yaml
```

## 🖼️ Multimodal Examples

### 3. LLaVA-style Image-Text Model

**Config `image_text.yaml`:**
```yaml
# Base LLM
model_name_or_path: "microsoft/DialoGPT-medium"

# Vision encoder
vision_encoder: "openai/clip-vit-base-patch32"
projection_type: "mlp"

# Dataset with images
dataset: ["liuhaotian/LLaVA-Instruct-150K"]

# Modalities
modalities:
  input: ["image", "text"]
  output: ["text"]

# Multimodal training settings
freeze_vision_encoder: true
freeze_llm: false
train_projection_only: false

# Standard training parameters
per_device_train_batch_size: 2
num_train_epochs: 3
learning_rate: 2e-5
max_seq_length: 2048
output_dir: "./llava_model"

# LoRA for efficiency
use_peft: true
lora:
  r: 64
  alpha: 128
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# Dataset fields
conversation_field: "conversations"
image_field: "image"
```

### 4. Audio-Text Model Training

**Config `audio_text.yaml`:**
```yaml
model_name_or_path: "microsoft/DialoGPT-medium"
audio_encoder: "openai/whisper-base"
projection_type: "linear"

dataset: ["speech_dataset"]

modalities:
  input: ["audio", "text"]
  output: ["text"]

freeze_audio_encoder: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
output_dir: "./audio_text_model"
```

## 🚀 Distributed Training

### 5. Multi-GPU Training with FSDP

**Run with Accelerate:**
```bash
# Create accelerate config (first time only)
poetry run accelerate config

# Or use our pre-configured FSDP config
PYTHONPATH="${PYTHONPATH}:src/" poetry run accelerate launch \
    --config_file accelerate/fsdp_config.yaml \
    scripts/train_multimodal.py \
    your_config.yaml
```

### 6. Training with DeepSpeed

```bash
PYTHONPATH="${PYTHONPATH}:src/" poetry run accelerate launch \
    --config_file accelerate/deepspeed_config.yaml \
    scripts/train_multimodal.py \
    your_config.yaml
```

## 📊 Monitoring and Logging

### 7. Training with Weights & Biases

**Add to config:**
```yaml
# ... your other parameters ...

# W&B logging
logging_strategy: "steps"
logging_steps: 50
report_to: ["wandb"]

# W&B project settings
wandb_project: "my_multimodal_training"
wandb_run_name: "experiment_1"
wandb_tags: ["multimodal", "llava", "test"]
```

### 8. TensorBoard Logging

```yaml
# ... your other parameters ...

report_to: ["tensorboard"]
logging_dir: "./logs"
```

**View logs:**
```bash
poetry run tensorboard --logdir ./logs
```

## 🎯 Practical Tips

### Configuration Structure
```yaml
# Essential parameters
model_name_or_path: "..."      # Any HuggingFace model
dataset: [...]                 # List of datasets

# Modality settings (for multimodal)
modalities:
  input: [...]                 # Input modalities
  output: [...]                # Output modalities

# Training parameters (standard HF arguments)
per_device_train_batch_size: 2
num_train_epochs: 3
learning_rate: 2e-5
output_dir: "./my_model"

# Efficiency settings
use_peft: true                 # Enable LoRA
gradient_accumulation_steps: 4
gradient_checkpointing: true
dataloader_num_workers: 4

# Memory optimization
fp16: true                     # or bf16: true
remove_unused_columns: false
```

### Memory Optimization

**For Large Models:**
```yaml
# Use LoRA
use_peft: true
lora:
  r: 16  # smaller rank = less memory

# Reduce batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 16

# Enable optimizations
gradient_checkpointing: true
fp16: true  # or bf16: true
dataloader_num_workers: 0
```

**For Small GPUs:**
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
max_seq_length: 1024  # shorter sequences
freeze_vision_encoder: true  # freeze encoders
train_projection_only: true  # train only projections
```

## 🔧 Debugging

### Check Configuration
```bash
# Check if config is valid
poetry run python -c "
from src.any2any_trainer.utils.config import load_config
config = load_config('your_config.yaml')
print('Config is valid!')
print(f'Model: {config.model_name_or_path}')
print(f'Batch size: {config.per_device_train_batch_size}')
"
```

### Test Data Loading
```bash
# Test that dataset loads correctly
poetry run python -c "
from datasets import load_dataset
ds = load_dataset('tatsu-lab/alpaca', split='train[:10]')
print('Dataset loaded successfully!')
print(f'First example: {ds[0]}')
"
```

### Quick Validation Run
```yaml
# Add to your config for quick testing
max_steps: 10               # Train for only 10 steps
eval_steps: 5               # Evaluate every 5 steps
save_steps: 5               # Save every 5 steps
logging_steps: 1            # Log every step
```

## ✅ Next Steps

1. **Read the detailed documentation:**
   - [HF_MODELS_USAGE.md](HF_MODELS_USAGE.md) - In-depth HuggingFace models guide
   - [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Library architecture

2. **Explore example configurations:**
   - `configs/sft/` - Supervised fine-tuning examples
   - `configs/any2any/` - Any-to-any model examples

3. **Customize for your needs:**
   - Create your dataset loaders
   - Add custom model architectures
   - Implement new modalities

Good luck with your training! 🚀 