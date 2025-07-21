"""
Dataset loading functionality for Any2Any Trainer.
"""

import os
from typing import Tuple, Optional
from datasets import load_dataset as hf_load_dataset, Dataset
from ..utils.config import TrainingConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


def load_dataset(config: TrainingConfig) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load dataset according to configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info(f"📊 Loading dataset: {config.dataset}")
    
    if not config.dataset:
        raise ValueError("No dataset specified in configuration")
    
    # Handle dataset configuration
    if len(config.dataset) >= 2:
        # Format: ["dataset_name", "config_name"]
        dataset_name = config.dataset[0]
        dataset_config = config.dataset[1]
        logger.info(f"📋 Using dataset config: {dataset_config}")
    else:
        # Format: ["dataset_name"]
        dataset_name = config.dataset[0]
        dataset_config = None
    
    try:
        # Проверяем тип датасета и загружаем соответственно
        if os.path.exists(dataset_name):
            if os.path.isdir(dataset_name):
                # HuggingFace датасет на диске
                logger.info(f"📁 Loading HF dataset from disk: {dataset_name}")
                from datasets import load_from_disk
                dataset = load_from_disk(dataset_name)
            elif dataset_name.endswith('.jsonl'):
                # JSONL файл
                logger.info(f"📁 Loading local JSONL file: {dataset_name}")
                dataset = hf_load_dataset('json', data_files=dataset_name)
            else:
                raise ValueError(f"Unsupported local file format: {dataset_name}")
        else:
            # Load dataset from HuggingFace Hub
            if dataset_config:
                dataset = hf_load_dataset(dataset_name, dataset_config)
            else:
                dataset = hf_load_dataset(dataset_name)
        
        # Обработка датасета в зависимости от типа
        from datasets import Dataset, DatasetDict
        
        if isinstance(dataset, Dataset):
            # Прямой Dataset (из load_from_disk)
            logger.info(f"📋 Direct dataset loaded, size: {len(dataset)}")
            train_dataset = dataset
            eval_dataset = None
            
            # Создаем eval split если датасет достаточно большой
            if len(train_dataset) > 10:
                split_dataset = train_dataset.train_test_split(test_size=0.1)
                train_dataset = split_dataset["train"]
                eval_dataset = split_dataset["test"]
                logger.info(f"📋 Split dataset: train={len(train_dataset)}, test={len(eval_dataset)}")
        
        elif isinstance(dataset, DatasetDict):
            # DatasetDict с несколькими splits
            train_split_candidates = ["train", "train_sft", "training"]
            train_dataset = None
            
            for split_name in train_split_candidates:
                if split_name in dataset:
                    train_dataset = dataset[split_name]
                    logger.info(f"📋 Using train split: {split_name}")
                    break
            
            if train_dataset is None:
                # Use first available split
                split_name = list(dataset.keys())[0]
                train_dataset = dataset[split_name]
                logger.warning(f"⚠️ No standard train split found, using '{split_name}'")
        
            # Try to get validation split
            eval_split_candidates = ["validation", "val", "test", "test_sft", "eval"]
            eval_dataset = None
            
            for split_name in eval_split_candidates:
                if split_name in dataset:
                    eval_dataset = dataset[split_name]
                    logger.info(f"📋 Using eval split: {split_name}")
                    break
            else:
                # Split train dataset
                if len(train_dataset) > 100:
                    split_dataset = train_dataset.train_test_split(test_size=0.1)
                    train_dataset = split_dataset["train"]
                    eval_dataset = split_dataset["test"]
        
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")
        
        # Validate dataset format - expect standard conversation format
        def validate_conversation_format(example):
            """Validate that dataset follows expected conversation format."""
            # Expected format: conversation field with list of messages
            conversation_field = config.conversation_field
            
            if conversation_field not in example:
                logger.debug(f"⚠️ Missing field '{conversation_field}' in example")
                return False
            
            conversations = example[conversation_field]
            if not isinstance(conversations, list) or len(conversations) == 0:
                return False
            
            # Validate conversation structure
            for msg in conversations:
                if not isinstance(msg, dict):
                    return False
                if "role" not in msg or "content" not in msg:
                    return False
                content = str(msg["content"]).strip()
                if not content or len(content) < 5:
                    return False
            
            return True
        
        # Apply validation
        original_train_size = len(train_dataset)
        train_dataset = train_dataset.filter(validate_conversation_format)
        logger.info(f"📊 Validated train dataset: {original_train_size} → {len(train_dataset)} examples")
        
        if eval_dataset:
            original_eval_size = len(eval_dataset)
            eval_dataset = eval_dataset.filter(validate_conversation_format)
            logger.info(f"📊 Validated eval dataset: {original_eval_size} → {len(eval_dataset)} examples")
        
        logger.info(f"✅ Loaded {len(train_dataset)} training examples")
        if eval_dataset:
            logger.info(f"✅ Loaded {len(eval_dataset)} evaluation examples")
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        logger.error(f"❌ Failed to load dataset {dataset_name}: {e}")
        raise 