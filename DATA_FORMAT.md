# 📋 Data Format Specification for Any2Any Trainer

## 🎯 Standard Format

Any2Any Trainer использует **единый стандартный формат данных** для всех типов обучения. Мы НЕ предоставляем автоматическую конвертацию различных форматов — пользователь должен привести свои данные к этому стандарту.

### Обязательный формат: OpenAI-style Conversations

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

## 🔧 Конфигурация

В YAML конфиге вы можете указать имя поля с разговорами:

```yaml
conversation_field: "conversations"  # по умолчанию
# или
conversation_field: "messages"      # если ваше поле называется иначе
```

## 📚 Поддерживаемые роли

- `"user"` — сообщения пользователя
- `"assistant"` — ответы модели
- `"system"` — системные промпты
- любые другие роли (будут обработаны как есть)

## 🎪 Примеры разных типов данных

### 1. Простой Q&A
```json
{
  "conversations": [
    {"role": "user", "content": "Что такое PyTorch?"},
    {"role": "assistant", "content": "PyTorch — это библиотека машинного обучения..."}
  ]
}
```

### 2. Мультимодальные данные
```json
{
  "conversations": [
    {
      "role": "user", 
      "content": "Опиши эту картинку",
      "image": "path/to/image.jpg"
    },
    {
      "role": "assistant",
      "content": "На изображении видна кошка..."
    }
  ]
}
```

### 3. Многоходовой диалог
```json
{
  "conversations": [
    {"role": "user", "content": "Привет! Как дела?"},
    {"role": "assistant", "content": "Привет! Всё отлично, спасибо!"},
    {"role": "user", "content": "Можешь помочь с Python?"},
    {"role": "assistant", "content": "Конечно! Что именно нужно?"}
  ]
}
```

### 4. С системным промптом
```json
{
  "conversations": [
    {"role": "system", "content": "Ты полезный ассистент-программист"},
    {"role": "user", "content": "Как написать цикл в Python?"},
    {"role": "assistant", "content": "В Python есть несколько видов циклов..."}
  ]
}
```

## ⚡ Any-to-Any расширения

Для мультимодальных данных вы можете добавлять дополнительные поля:

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Опиши видео", 
      "video": "path/to/video.mp4"
    },
    {
      "role": "assistant",
      "content": "Сгенерированное описание...",
      "audio": "path/to/speech.wav"
    }
  ]
}
```

## 🚫 Что НЕ поддерживается автоматически

Мы НЕ конвертируем автоматически:
- ❌ Alpaca format (`instruction`, `input`, `output`)
- ❌ ShareGPT format 
- ❌ Plain text datasets
- ❌ Custom JSON structures

**Если ваш датасет в другом формате — преобразуйте его заранее!**

## 🛠️ Как преобразовать ваши данные

### Из Alpaca в Conversations
```python
def alpaca_to_conversations(example):
    conversations = [
        {"role": "user", "content": example["instruction"]}
    ]
    if example.get("input"):
        conversations[0]["content"] += f"\n\nДополнительная информация: {example['input']}"
    
    conversations.append({
        "role": "assistant", 
        "content": example["output"]
    })
    
    return {"conversations": conversations}

# Применить к датасету
dataset = dataset.map(alpaca_to_conversations)
```

### Из plain text в Conversations
```python
def text_to_conversations(example):
    return {
        "conversations": [
            {"role": "user", "content": "Продолжи текст:"},
            {"role": "assistant", "content": example["text"]}
        ]
    }

dataset = dataset.map(text_to_conversations)
```

## 📖 Совместимость с HuggingFace

Этот формат совместим с:
- ✅ [OpenAI ChatCompletion API](https://platform.openai.com/docs/api-reference/chat)
- ✅ [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/chat_templating)
- ✅ [TRL ChatML format](https://huggingface.co/docs/trl/sft_trainer#format-your-input-prompts)
- ✅ [effective_llm_alignment](https://github.com/VikhrModels/effective_llm_alignment)

## 🎯 Рекомендации

1. **Используйте стандартный формат** — не придумывайте свой
2. **Преобразуйте данные заранее** — не полагайтесь на автоконвертацию
3. **Проверьте структуру** — убедитесь что есть поля `role` и `content`
4. **Используйте осмысленные роли** — `user`, `assistant`, `system`

---

**💡 Принцип**: Один стандартный формат для всех задач. Простота > Универсальность. 