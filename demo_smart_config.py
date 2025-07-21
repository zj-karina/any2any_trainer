#!/usr/bin/env python3
"""
🎯 Демонстрация умной конфигурации Any2Any Trainer

Показывает как теперь пользователи могут просто указать модальности
и система автоматически определит тип модели!
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from any2any_trainer.utils.config import TrainingConfig
import yaml


def demo_smart_configs():
    """Демонстрирует умные конфигурации."""
    print("🎯 Демонстрация умной конфигурации Any2Any Trainer\n")
    
    # 1. Простая LLM (text → text)
    print("1️⃣ Простая LLM конфигурация:")
    simple_config = {
        "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "modalities": {"input": ["text"], "output": ["text"]},
        "dataset": ["HuggingFaceH4/ultrachat_200k"],
        "output_dir": "./output/simple_llm",
        "use_peft": True,
        "lora": {"r": 32, "alpha": 64}
    }
    
    config = TrainingConfig(**simple_config)
    print(f"   📋 Модальности: {config.modalities}")
    print(f"   🤖 Автоопределен: model_type = '{config.model_type}'\n")
    
    # 2. Мультимодальная модель (image + text → text)
    print("2️⃣ Мультимодальная конфигурация (LLaVA-style):")
    multimodal_config = {
        "model_name_or_path": "microsoft/DialoGPT-medium",
        "modalities": {"input": ["image", "text"], "output": ["text"]},
        "encoders": {
            "image": {"model": "openai/clip-vit-large-patch14", "freeze": True}
        },
        "dataset": ["llava_conversations.jsonl"],
        "output_dir": "./output/multimodal_model"
    }
    
    config = TrainingConfig(**multimodal_config)
    print(f"   📋 Модальности: {config.modalities}")
    print(f"   🤖 Автоопределен: model_type = '{config.model_type}'\n")
    
    # 3. Any2Any модель (все модальности)
    print("3️⃣ Any2Any конфигурация:")
    any2any_config = {
        "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "modalities": {
            "input": ["text", "image", "audio"], 
            "output": ["text", "image", "audio"]
        },
        "encoders": {
            "image": {"model": "openai/clip-vit-large-patch14", "freeze": True},
            "audio": {"model": "openai/whisper-base", "freeze": True}
        },
        "decoders": {
            "image": {"model": "stabilityai/stable-diffusion-2-1"},
            "audio": {"model": "microsoft/speecht5_tts"}
        },
        "dataset": ["custom/multimodal_conversations"],
        "output_dir": "./output/any2any_model"
    }
    
    config = TrainingConfig(**any2any_config)
    print(f"   📋 Модальности: {config.modalities}")
    print(f"   🤖 Автоопределен: model_type = '{config.model_type}'\n")


def show_yaml_examples():
    """Показывает примеры YAML конфигураций."""
    print("📝 Примеры YAML конфигураций:\n")
    
    print("1️⃣ Простая LLM (configs/sft/simple_modality_flow.yaml):")
    simple_yaml = """
model_name_or_path: "Qwen/Qwen2.5-1.5B-Instruct"
# model_type автоматически определится как "standard"!
modalities:
  input: ["text"]
  output: ["text"]
dataset: ["HuggingFaceH4/ultrachat_200k"]
use_peft: true
lora:
  r: 32
  alpha: 64
    """
    print(simple_yaml)
    
    print("2️⃣ Мультимодальная модель (configs/sft/multimodal_flow.yaml):")
    multimodal_yaml = """
model_name_or_path: "microsoft/DialoGPT-medium"
# model_type автоматически определится как "multimodal"!
modalities:
  input: ["image", "text"]
  output: ["text"]
encoders:
  image:
    model: "openai/clip-vit-large-patch14"
    freeze: true
dataset: ["llava_conversations.jsonl"]
    """
    print(multimodal_yaml)
    
    print("3️⃣ Any2Any модель (configs/sft/any2any_flow.yaml):")
    any2any_yaml = """
model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
# model_type автоматически определится как "any2any"!
modalities:
  input: ["text", "image", "audio"]
  output: ["text", "image", "audio"]
encoders:
  image:
    model: "openai/clip-vit-large-patch14"
    freeze: true
  audio:
    model: "openai/whisper-base"
    freeze: true
decoders:
  image:
    model: "stabilityai/stable-diffusion-2-1"
  audio:
    model: "microsoft/speecht5_tts"
    """
    print(any2any_yaml)


def show_auto_detection_logic():
    """Объясняет логику автоопределения."""
    print("🧠 Логика автоопределения model_type:\n")
    
    rules = [
        ("🔤 text → text", "standard", "Обычная языковая модель"),
        ("🖼️ image → image", "standard", "Однотипная модальность"),
        ("🖼️📝 image+text → text", "multimodal", "Мультимодальный вход, текстовый выход"),
        ("🖼️ image → text", "multimodal", "Vision-to-text модель"),
        ("📝🖼️🔊 text+image+audio → text", "multimodal", "Множественный вход, текстовый выход"),
        ("📝 text → text+image", "any2any", "Множественный выход"),
        ("🖼️🔊 image+audio → text+audio", "any2any", "Сложные комбинации")
    ]
    
    for modalities, model_type, description in rules:
        print(f"   {modalities:<25} → {model_type:<12} | {description}")
    
    print()


def main():
    """Главная функция демонстрации."""
    print("=" * 80)
    print("🎯 Any2Any Trainer - Умная конфигурация")
    print("=" * 80)
    print()
    
    show_auto_detection_logic()
    demo_smart_configs()
    show_yaml_examples()
    
    print("✨ Преимущества умной конфигурации:")
    print("   • Не нужно знать и указывать model_type")
    print("   • Меньше ошибок в конфигурации")
    print("   • Простой и интуитивный синтаксис")
    print("   • Автоматическая валидация")
    print()
    
    print("🚀 Запуск обучения:")
    print("   poetry run python scripts/train_multimodal.py configs/sft/simple_modality_flow.yaml")
    print("   poetry run python scripts/train_multimodal.py configs/sft/multimodal_flow.yaml")
    print("   poetry run python scripts/train_multimodal.py configs/sft/any2any_flow.yaml")
    print()
    
    print("=" * 80)


if __name__ == "__main__":
    main() 