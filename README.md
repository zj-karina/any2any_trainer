# 🚀 Any2Any Trainer - Universal Multimodal Training Toolkit

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

## 🚀 How to Use

### 📦 Installation

**Quick Setup:**
```bash
cd any2any_trainer
poetry install
export HF_HOME=/mnt/hf/
```

## 🚀 Quick Start

```bash
# Быстрый тест (самый простой)
poetry run python scripts/train_multimodal.py configs/sft/minimal_working.yaml

# Основной тест функциональности
poetry run python test_installation.py  # Confirms: Model loading, LoRA, CUDA, forward pass
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


```json
{
  "conversations": [
    {
      "role": "user", 
      "content": "What is machine learning?"
    },
    {
      "role": "assistant",
      "content": "Machine learning is a subset of artificial intelligence..."
    }
  ]
}
```

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