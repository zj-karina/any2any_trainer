# 🤗 HuggingFace Models Usage Guide

This guide explains how to use any HuggingFace model with any2any_trainer without complex modifications.

## 🎯 Core Philosophy

**"Simple things should be simple, complex things should be possible"**

- ✅ Use any model from HuggingFace Hub directly
- ✅ Minimal configuration for standard training  
- ✅ Flexible architecture for complex multimodal scenarios
- ✅ No unnecessary abstractions or wrappers

## 🚀 Simple Usage Examples

### 1. Training Any LLM

```yaml
# Just specify the model!
model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
dataset: ["tatsu-lab/alpaca"]
per_device_train_batch_size: 2
output_dir: "./my_model"
```

**Supported models:**
- **Llama family:** `meta-llama/Llama-2-7b-hf`, `NousResearch/Llama-2-7b-chat-hf`
- **Qwen family:** `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-7B-Instruct`
- **Mistral family:** `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.2`
- **Phi family:** `microsoft/phi-2`, `microsoft/Phi-3-mini-4k-instruct`
- **Gemma family:** `google/gemma-7b`, `google/gemma-7b-it`
- **And many others!**

### 2. Efficient Training with LoRA

```yaml
model_name_or_path: "meta-llama/Llama-2-7b-chat-hf"
dataset: ["tatsu-lab/alpaca"]

# LoRA configuration
use_peft: true
lora:
  r: 64
  alpha: 128
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  dropout: 0.05

per_device_train_batch_size: 4
gradient_accumulation_steps: 4
num_train_epochs: 3
```

### 3. QLoRA (4-bit quantization)

```yaml
model_name_or_path: "meta-llama/Llama-2-13b-chat-hf"
dataset: ["tatsu-lab/alpaca"]

# QLoRA settings
use_peft: true
load_in_4bit: true
bnb_4bit_quant_type: "nf4"
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: "bf16"

lora:
  r: 64
  alpha: 16
  target_modules: ["q_proj", "v_proj"]
  dropout: 0.1

per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

## 🖼️ Multimodal Models

### 1. Vision-Language Models (LLaVA style)

```yaml
# Base language model
model_name_or_path: "microsoft/DialoGPT-medium"

# Vision encoder (any CLIP-compatible model)
vision_encoder: "openai/clip-vit-base-patch32"
# Alternative encoders:
# vision_encoder: "google/siglip-base-patch16-224"
# vision_encoder: "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

# Projection layer type
projection_type: "mlp"  # Options: linear, mlp, transformer

# Dataset
dataset: ["liuhaotian/LLaVA-Instruct-150K"]

# Training settings
freeze_vision_encoder: true  # Usually keep vision encoder frozen
freeze_llm: false           # Train the LLM
train_projection_only: false # Train projection + LLM

modalities:
  input: ["image", "text"]
  output: ["text"]
```

### 2. Audio-Language Models

```yaml
model_name_or_path: "microsoft/DialoGPT-medium"

# Audio encoder
audio_encoder: "openai/whisper-base"
# Alternative encoders:
# audio_encoder: "facebook/wav2vec2-base-960h"
# audio_encoder: "microsoft/wavlm-base"

projection_type: "linear"

dataset: ["speech_dataset"]

freeze_audio_encoder: true
modalities:
  input: ["audio", "text"]
  output: ["text"]
```

### 3. Video-Language Models

```yaml
model_name_or_path: "microsoft/DialoGPT-medium"

# Video encoder
video_encoder: "MCG-NJU/videomae-base"
# Alternative encoders:
# video_encoder: "facebook/timesformer-base-finetuned-k400"

projection_type: "transformer"

dataset: ["video_dataset"]

freeze_video_encoder: true
modalities:
  input: ["video", "text"]
  output: ["text"]
```

## 🔧 Advanced Configuration

### 1. Multiple LoRA Adapters

```yaml
model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"

use_peft: true
lora:
  r: 64
  alpha: 128
  # Target all linear layers
  target_modules: 
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  dropout: 0.05
  
  # Advanced LoRA settings
  use_rslora: true      # Rank Stabilized LoRA
  use_dora: false       # DoRA (Weight-Decomposed Low-Rank Adaptation)
  lora_alpha: 128
  lora_dropout: 0.05
```

### 2. Custom Model Loading

```yaml
model_name_or_path: "your-custom/model"

# Model loading parameters
model_kwargs:
  torch_dtype: "auto"
  trust_remote_code: true
  device_map: "auto"
  low_cpu_mem_usage: true

# Tokenizer parameters
tokenizer_kwargs:
  padding_side: "right"
  truncation_side: "right"
  trust_remote_code: true
```

### 3. Multiple Datasets

```yaml
model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"

# Multiple datasets with different formats
dataset:
  - "tatsu-lab/alpaca"
  - "Open-Orca/OpenOrca"
  - "teknium/GPT4-LLM-Cleaned"

# Dataset processing
dataset_config:
  max_seq_length: 2048
  packing: true              # Pack multiple examples into one sequence
  remove_unused_columns: false
  
# Custom dataset field mapping
dataset_field_mapping:
  instruction_field: "instruction"
  input_field: "input"
  output_field: "output"
  conversation_field: "conversations"
```

## ⚡ Performance Optimization

### 1. Memory-Efficient Training

```yaml
model_name_or_path: "meta-llama/Llama-2-7b-hf"

# Memory optimizations
gradient_checkpointing: true
dataloader_num_workers: 4
remove_unused_columns: false
dataloader_pin_memory: true

# Precision settings
fp16: true                    # Use for older GPUs
# bf16: true                  # Use for modern GPUs (A100, H100)

# Batch size optimization
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
max_grad_norm: 1.0

# DeepSpeed settings
deepspeed: "deepspeed_configs/zero2_config.json"
```

### 2. Fast Training Settings

```yaml
model_name_or_path: "microsoft/DialoGPT-small"

# Training efficiency
max_seq_length: 1024          # Shorter sequences = faster
num_train_epochs: 1
learning_rate: 5e-5           # Higher LR for faster convergence

# Disable slow operations
evaluation_strategy: "no"     # Skip evaluation during training
save_strategy: "epoch"        # Save less frequently
logging_steps: 100            # Log less frequently

# Use more workers
dataloader_num_workers: 8
```

## 🎛️ Model-Specific Settings

### 1. Llama Models

```yaml
model_name_or_path: "meta-llama/Llama-2-7b-chat-hf"

# Llama-specific settings
use_auth_token: true          # Required for Llama models

model_kwargs:
  torch_dtype: "float16"
  device_map: "auto"
  
# LoRA targets for Llama
lora:
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### 2. Qwen Models

```yaml
model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"

# Qwen-specific settings
model_kwargs:
  torch_dtype: "bfloat16"     # Qwen works well with bf16
  trust_remote_code: true     # Required for some Qwen models

# LoRA targets for Qwen
lora:
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### 3. Mistral Models

```yaml
model_name_or_path: "mistralai/Mistral-7B-Instruct-v0.2"

# Mistral-specific settings
model_kwargs:
  torch_dtype: "float16"
  device_map: "auto"
  
# LoRA targets for Mistral
lora:
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

## 📊 Data Format Examples

### 1. Standard Instruction Format

```json
{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "The capital of France is Paris."
}
```

### 2. Conversational Format

```json
{
  "conversations": [
    {"role": "user", "content": "Hello! How are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"}
  ]
}
```

### 3. Multimodal Format

```json
{
  "conversations": [
    {
      "role": "user", 
      "content": "What's in this image?",
      "image": "path/to/image.jpg"
    },
    {
      "role": "assistant",
      "content": "I can see a beautiful sunset over the ocean."
    }
  ]
}
```

## 🚨 Common Issues and Solutions

### 1. Out of Memory (OOM)

**Solution:**
```yaml
# Reduce batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 32

# Use LoRA
use_peft: true
lora:
  r: 16  # Smaller rank

# Enable optimizations
gradient_checkpointing: true
fp16: true

# Use DeepSpeed
deepspeed: "deepspeed_configs/zero3_config.json"
```

### 2. Slow Training

**Solution:**
```yaml
# Increase batch size (if memory allows)
per_device_train_batch_size: 8
gradient_accumulation_steps: 4

# Use more workers
dataloader_num_workers: 8

# Enable Flash Attention (if available)
attn_implementation: "flash_attention_2"

# Use bf16 for modern GPUs
bf16: true
```

### 3. Model Loading Issues

**Solution:**
```yaml
# Add trust_remote_code for custom models
model_kwargs:
  trust_remote_code: true

# Use auth token for gated models
use_auth_token: true

# Specify torch_dtype explicitly
model_kwargs:
  torch_dtype: "auto"
  device_map: "auto"
```

## 🎯 Best Practices

### 1. Choosing Model Size

- **Small models (1-3B):** Good for experimentation, fast iteration
- **Medium models (7-13B):** Good balance of quality and speed
- **Large models (70B+):** Best quality, require significant resources

### 2. LoRA Configuration

- **Small tasks:** `r=16, alpha=32`
- **Standard tasks:** `r=64, alpha=128`
- **Complex tasks:** `r=128, alpha=256`

### 3. Learning Rate

- **Full fine-tuning:** `1e-5 to 5e-5`
- **LoRA:** `1e-4 to 5e-4`
- **QLoRA:** `2e-4 to 1e-3`

### 4. Batch Size

- **Large models:** Start with batch size 1, increase `gradient_accumulation_steps`
- **Small models:** Can use larger batch sizes (4-16)
- **Target effective batch size:** 64-256 for most tasks

## ✅ Ready-to-Use Examples

### Complete Configuration for Llama-2 7B with LoRA

```yaml
# Llama-2 7B with LoRA - ready to run!
model_name_or_path: "meta-llama/Llama-2-7b-chat-hf"
dataset: ["tatsu-lab/alpaca"]

use_peft: true
lora:
  r: 64
  alpha: 128
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  dropout: 0.05

per_device_train_batch_size: 2
gradient_accumulation_steps: 8
num_train_epochs: 3
learning_rate: 2e-4
max_seq_length: 2048

# Optimizations
gradient_checkpointing: true
fp16: true
dataloader_num_workers: 4

# Logging
logging_steps: 50
save_strategy: "epoch"
output_dir: "./llama2_7b_lora"

# Optional: W&B logging
report_to: ["wandb"]
wandb_project: "llama2_training"
```

Save this as `llama2_example.yaml` and run:

```bash
PYTHONPATH="${PYTHONPATH}:src/" poetry run python \
    scripts/train_multimodal.py \
    llama2_example.yaml
```

That's it! 🚀 