"""
Prediction function for Hangman model.
"""
import torch
import torch.nn.functional as F
import string
from typing import List, Dict, Any, Union, Optional
import os

from simple_dataloader import HangmanTokenizer
from model import create_model
from safetensors.torch import load_file

# Initialize tokenizer and model
tokenizer = HangmanTokenizer()
model_path = "checkpoints/checkpoint-57500"
model = create_model('pretrained_bert', {"vocab_size": tokenizer.vocab_size})

# Set up device and load model weights
device = "cuda" if torch.cuda.is_available() else "cpu"
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    try:
        # Load the model using safetensors
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))

        # Move the state dict to the correct device
        for key, tensor in state_dict.items():
            state_dict[key] = tensor.to(device)

        # Load the state dict into the model
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load model: {e}")
else:
    print(f"Warning: Model path {model_path} not found, using random weights")
    
model.eval()
model.to(device)
print(device)


def predict_next_letter(
    guessed_letters: List[str],
    current_state: List[str]
) -> Dict[str, float]:
    """
    Predict the next letter for a Hangman game state.
    
    Args:
        guessed_letters: List of already guessed letters.
        current_state: Current word state with revealed letters and '*' for unknowns.
        
    Returns:
        Dictionary mapping letters to their probabilities.
    """
    # Create input sequence: [guessed_letters, SEP, state]
    sequence = list(guessed_letters) + ['[SEP]'] + list(current_state)
    
    # Encode sequence
    input_ids = tokenizer.encode(sequence)
    
    # Create tensor
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Get probabilities
        if isinstance(outputs, dict):
            probs = outputs['probs']
        else:
            probs = outputs
    
    # Convert to numpy
    probs = probs.squeeze().cpu().numpy()
    
    # Create results dictionary
    results = {}
    
    # Fill results with all letters and their probabilities
    for i, letter in enumerate(string.ascii_lowercase):
        results[letter] = float(probs[i])
        
        # Zero out already guessed letters
        if letter in guessed_letters:
            results[letter] = 0.0
    
    return results


def get_top_predictions(
    guessed_letters: List[str],
    current_state: List[str],
    top_n: int = 5
) -> List[Dict[str, Union[str, float]]]:
    """
    Get the top N letter predictions for a Hangman game state.
    
    Args:
        guessed_letters: List of already guessed letters.
        current_state: Current word state with revealed letters and '*' for unknowns.
        top_n: Number of top predictions to return.
        
    Returns:
        List of dictionaries with letter and probability pairs, sorted by probability.
    """
    # Get all letter probabilities
    letter_probs = predict_next_letter(guessed_letters, current_state)
    
    # Sort by probability
    sorted_probs = sorted(letter_probs.items(), key=lambda x: x[1], reverse=True)
    
    # Format the result
    return [{"letter": letter, "probability": prob} for letter, prob in sorted_probs[:top_n]]