"""
Script to run the Hangman trainer without using wandb for logging.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import TrainingArguments, Trainer

from dataloader import HangmanTokenizer, create_datasets
from model import create_model

max_steps = 200000

# Set configuration variables
cache_dir = 'cache'
model_type = 'pretrained_bert'
max_seq_length = 50
batch_size = 256
learning_rate = 1e-4
seed = 42
checkpoint_dir = 'checkpoints'
log_dir = 'logs'
resume_from_checkpoint = None  # Start training from scratch

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

# Create directories
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Create tokenizer and datasets
tokenizer = HangmanTokenizer()
train_dataset, val_dataset, test_dataset = create_datasets(
    tokenizer,
    max_seq_length
)

# Create model
model_config = {
    'vocab_size': tokenizer.vocab_size
}
model = create_model(model_type, model_config)
model = model.to(device)

# Make all parameters contiguous
for param in model.parameters():
    if not param.is_contiguous():
        param.data = param.data.contiguous()

# Custom data collator that knows how to handle our dataset format
def custom_data_collator(features):
    """
    Custom data collator that handles our dataset that returns (input_ids, target_masks) tuples.
    """
    if isinstance(features[0], tuple) and len(features[0]) == 2:
        # Unzip the batch of (input_ids, target_masks) tuples
        input_ids, target_masks = zip(*features)
        
        # Stack tensors
        input_ids = torch.stack(list(input_ids))
        target_masks = torch.stack(list(target_masks))
        
        return {
            "input_ids": input_ids,
            "target_masks": target_masks  # Change from "labels" to "target_masks"
        }
    else:
        # Fallback to default handling
        return {key: torch.stack([example[key] for example in features]) 
                for key in features[0].keys()}

# Define training arguments WITHOUT wandb
training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=0.01,
    logging_dir=log_dir,
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    max_steps=max_steps,
    seed=seed,
    disable_tqdm=False,
    report_to="none",  # Explicitly disable wandb
    save_total_limit=10,  # Keep only 10 models saved
)

# Create a custom trainer class
class HangmanTrainer(Trainer):
    """
    Custom Trainer for Hangman model.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract input_ids and target_masks from inputs
        input_ids = inputs.get("input_ids")
        target_masks = inputs.get("target_masks")  # Change from "labels" to "target_masks"
        
        # If inputs come from our dataloader format, adapt accordingly
        if input_ids is None and len(inputs) == 2 and isinstance(inputs, tuple):
            input_ids, target_masks = inputs
        elif isinstance(inputs, dict):
            # Make a copy of inputs dict without target_masks to pass to model
            model_inputs = {k: v for k, v in inputs.items() if k != "target_masks" and k != "labels"}  # Filter out both target_masks and labels
            input_ids = inputs["input_ids"]  # For reference only
        else:
            raise ValueError(f"Unexpected input type: {type(inputs)}")
        
        # Forward pass through the model - only pass input_ids, not labels
        outputs = model(input_ids) if not isinstance(inputs, dict) else model(**model_inputs)
        
        # Get logits from model output
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        # Use Binary Cross Entropy loss for multi-label classification
        # where targets are 0-1 and can be multiclass
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, target_masks.float())
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to handle our custom input format.
        """
        # Make sure we don't pass labels to the model
        has_labels = "target_masks" in inputs
        if has_labels:
            target_masks = inputs.pop("target_masks")
            if "labels" in inputs:
                inputs.pop("labels")
        
        # Pass only valid inputs to the model
        with torch.no_grad():
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            
        if has_labels:
            # Get the loss
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
                
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, target_masks.float())
            
            return (loss, logits, target_masks)
        
        return (None, outputs, None)
    
    def _save(self, output_dir: str, state_dict=None):
        """
        Override _save to make sure all tensors are contiguous before saving.
        """
        # Make sure all parameters are contiguous
        if state_dict is None:
            state_dict = self.model.state_dict()
            
        # Create contiguous copies of non-contiguous tensors
        contiguous_state_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and not v.is_contiguous():
                contiguous_state_dict[k] = v.contiguous()
            else:
                contiguous_state_dict[k] = v
                
        # Call the parent's _save with the contiguous state_dict
        super()._save(output_dir, contiguous_state_dict)

# Define a simple compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    # Convert probability distributions to letter indices
    pred_indices = np.argmax(logits, axis=1)
    
    # For multiclass classification, targets might be one-hot encoded or indices
    if labels.ndim > 1 and labels.shape[1] > 1:
        # If targets are one-hot encoded, get the indices of correct answers
        correct_targets = [np.where(labels[i] > 0)[0] for i in range(labels.shape[0])]
        # Check if prediction is one of the correct targets
        accuracy = np.mean([pred_indices[i] in correct_targets[i] for i in range(len(pred_indices))])
    else:
        # If targets are already indices
        target_indices = labels if labels.ndim == 1 else np.argmax(labels, axis=1)
        accuracy = np.mean(pred_indices == target_indices)
    
    return {
        'accuracy': float(accuracy),
    }

# Run the trainer
if __name__ == "__main__":
    print("Creating trainer...")
    trainer = HangmanTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=custom_data_collator,
    )
    
    # print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    # trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    print("Starting training from step 0...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(os.path.join(checkpoint_dir, "best_model"))
    
    print("Training completed.") 