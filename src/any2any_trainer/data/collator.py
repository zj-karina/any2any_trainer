"""
Data collator for multimodal training.
"""

from typing import Dict, List, Any
import torch
from ..utils.config import TrainingConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MultimodalCollator:
    """
    Data collator for multimodal models.
    
    Simple implementation for batching data.
    """
    
    def __init__(self, config: TrainingConfig, tokenizer=None):
        """
        Initialize collator.
        
        Args:
            config: Training configuration
            tokenizer: Tokenizer for text processing
        """
        self.config = config
        self.max_length = config.max_seq_length
        self.tokenizer = tokenizer
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into a batch.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Batched features
        """
        if not features:
            return {}
        
        # Process conversation data if tokenizer is available
        conversation_field = self.config.conversation_field
        if self.tokenizer and conversation_field in features[0]:
            # Convert conversations to text
            texts = []
            for feature in features:
                conversations = feature[conversation_field]
                if isinstance(conversations, list):
                    # Format conversations as chat
                    if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                        try:
                            text = self.tokenizer.apply_chat_template(
                                conversations, 
                                tokenize=False, 
                                add_generation_prompt=False
                            )
                            texts.append(text)
                        except Exception as e:
                            logger.debug(f"Chat template failed: {e}, using fallback")
                            # Fallback formatting
                            text_parts = []
                            for msg in conversations:
                                role = msg.get("role", "").strip()
                                content = msg.get("content", "").strip()
                                if role and content:
                                    text_parts.append(f"{role}: {content}")
                            texts.append("\n".join(text_parts))
                    else:
                        # Fallback formatting when no chat template
                        text_parts = []
                        for msg in conversations:
                            role = msg.get("role", "").strip()
                            content = msg.get("content", "").strip()
                            if role and content:
                                text_parts.append(f"{role}: {content}")
                        texts.append("\n".join(text_parts))
            
            # Filter out empty texts
            texts = [t for t in texts if t and len(t.strip()) > 0]
            if not texts:
                logger.warning("⚠️ All conversations are empty, creating dummy batch")
                return {
                    "input_ids": torch.tensor([[self.tokenizer.eos_token_id]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1]], dtype=torch.long),
                    "labels": torch.tensor([[-100]], dtype=torch.long),
                }
            
            # Tokenize texts
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Create batch with tokenized data
            batch = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
            
            # Add labels (same as input_ids for causal LM, but shifted)
            labels = tokenized["input_ids"].clone()
            
            # Check if we have valid sequences (length > 0)
            if labels.size(1) > 0:
                # For causal LM, we want to predict the next token, so shift labels
                if labels.size(1) > 1:
                    labels[:, :-1] = tokenized["input_ids"][:, 1:]
                    labels[:, -1] = -100  # Ignore last token in loss
                else:
                    # Single token sequence - just ignore it
                    labels[:, 0] = -100
            else:
                # Empty sequence - fill with ignore tokens
                labels.fill_(-100)
                
            batch["labels"] = labels
            
            logger.debug(f"📦 Created batch: input_ids shape={batch['input_ids'].shape}, labels shape={batch['labels'].shape}")
            
            return batch
        
        # Fallback: process existing tokenized data
        batch = {}
        
        for key in features[0].keys():
            values = [f[key] for f in features if key in f]
            
            if key in ["input_ids", "attention_mask", "labels"]:
                # Text fields - pad to max length
                if isinstance(values[0], list):
                    # Pad sequences
                    max_len = min(max(len(v) for v in values), self.max_length)
                    padded_values = []
                    
                    for v in values:
                        if len(v) > max_len:
                            v = v[:max_len]
                        else:
                            # Pad with 0 (or -100 for labels)
                            pad_value = -100 if key == "labels" else 0
                            v = v + [pad_value] * (max_len - len(v))
                        padded_values.append(v)
                    
                    batch[key] = torch.tensor(padded_values)
                else:
                    batch[key] = torch.stack(values)
            
            elif key in ["images", "audio", "video"]:
                # Media fields - stack if tensors
                if isinstance(values[0], torch.Tensor):
                    batch[key] = torch.stack(values)
                else:
                    # Keep as list for processing
                    batch[key] = values
            
            else:
                # Other fields - keep as list
                batch[key] = values
        
        return batch 