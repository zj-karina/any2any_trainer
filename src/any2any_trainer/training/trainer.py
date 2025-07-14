"""
Training functionality for multimodal models.

Simple implementation that extends HuggingFace Trainer.
"""

from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from datasets import Dataset

from ..utils.config import TrainingConfig, ConfigManager
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MultimodalTrainer(Trainer):
    """
    Multimodal trainer that extends HuggingFace Trainer.
    
    Simple implementation for training multimodal models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Any] = None,
        accelerator: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize the multimodal trainer.
        
        Args:
            model: The model to train
            config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            data_collator: Data collator for batching
            accelerator: Accelerate accelerator (optional)
        """
        self.config = config
        self.accelerator = accelerator
        
        # Convert config to HuggingFace TrainingArguments
        training_args = ConfigManager.to_training_arguments(config)
        
        # Initialize parent Trainer
        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            **kwargs
        )
        
        logger.info("🏋️ MultimodalTrainer initialized successfully")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training loss.
        
        Args:
            model: The model
            inputs: Model inputs
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch (passed by HF Trainer)
            
        Returns:
            Loss (and outputs if return_outputs=True)
        """
        # Ensure model is in training mode
        model.train()
        
        # Force all LoRA parameters to require gradients
        for name, param in model.named_parameters():
            if 'lora' in name.lower() or 'adapter' in name.lower():
                param.requires_grad_(True)
        
        # Simple implementation - just call the model
        outputs = model(**inputs)
            
        
        # Just return the loss from model outputs
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            # Fallback - compute loss manually
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            labels = inputs.get("labels")
            
            if labels is not None and logits is not None:
                # Standard causal LM loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            else:
                # Emergency fallback
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                if trainable_params:
                    loss = sum(p.sum() for p in trainable_params) * 0.0
                else:
                    loss = torch.tensor(0.0, requires_grad=True)
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Evaluate the model.
        
        Simple implementation that just computes loss.
        """
        logger.info("📊 Starting evaluation...")
        
        # Use parent's evaluate method
        eval_results = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        logger.info(f"✅ Evaluation completed. Loss: {eval_results.get('eval_loss', 'N/A')}")
        
        # Generate examples if configured
        if self.config.generate_eval_examples:
            try:
                self._generate_examples()
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate examples: {e}")
        
        return eval_results
    
    def _generate_examples(self, num_examples: int = 3):
        """
        Generate a few examples during evaluation.
        
        Simple implementation for monitoring training progress.
        """
        logger.info("🎯 Generating evaluation examples...")
        
        if not self.eval_dataset:
            logger.warning("No evaluation dataset available for example generation")
            return
        
        # Get a few examples from eval dataset
        examples = self.eval_dataset.select(range(min(num_examples, len(self.eval_dataset))))
        
        model = self.model
        model.eval()
        
        with torch.no_grad():
            for i, example in enumerate(examples):
                try:
                    # Simple text generation (assumes the model has a generate method)
                    if hasattr(model, 'generate') and "input_ids" in example:
                        input_ids = torch.tensor([example["input_ids"][:50]], device=model.device)  # First 50 tokens
                        
                        generated_ids = model.generate(
                            input_ids,
                            max_new_tokens=self.config.max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=self.tokenizer.eos_token_id if hasattr(self, 'tokenizer') else 0,
                        )
                        
                        # Decode generated text
                        if hasattr(self, 'tokenizer'):
                            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                            logger.info(f"Example {i+1}: {generated_text[:200]}...")
                        else:
                            logger.info(f"Example {i+1}: Generated {generated_ids.shape[1]} tokens")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate example {i+1}: {e}")
        
        model.train()
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save the model
            _internal_call: Internal flag
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        logger.info(f"💾 Saving model to {output_dir}")
        
        # Use parent's save method
        super().save_model(output_dir, _internal_call)
        
        # Also save the config
        try:
            import os
            from pathlib import Path
            config_path = Path(output_dir) / "training_config.yaml"
            ConfigManager.save_config(self.config, config_path)
            logger.info(f"✅ Training config saved to {config_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save training config: {e}")


class SimpleTrainer:
    """
    Even simpler trainer for basic use cases.
    
    Minimal implementation without all the HuggingFace complexity.
    """
    
    def __init__(self, model, tokenizer, config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
    
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, train_dataloader, num_epochs: int = 3):
        """Simple training loop."""
        logger.info(f"🎯 Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_dataloader:
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1
                
                if num_batches % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {num_batches}, Loss: {loss:.4f}")
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f"✅ Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        logger.info("🎉 Training completed!")
    
    def save_model(self, output_dir: str):
        """Save the model."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
        
        # Save tokenizer if available
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"💾 Model saved to {output_dir}") 