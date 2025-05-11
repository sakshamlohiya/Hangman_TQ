"""
Simple tokenizer and data loader for Hangman letter prediction model.
"""
import torch
import numpy as np
import random
import string
from typing import List, Dict, Tuple
import os

from utils import load_words, sample_game_states, prepare_model_input, sample_hard_game_states, build_hard_word_pool


# Load word files at initialization time
train_words = load_words('data/train_words.txt')
val_words = load_words('data/val_words.txt')
test_words = load_words('data/test_words.txt')

hard_words_train = build_hard_word_pool(train_words)
hard_words_val = build_hard_word_pool(val_words)
hard_words_test = build_hard_word_pool(test_words)

class HangmanTokenizer:
    """
    Tokenizer for Hangman game states.
    
    Handles conversion between characters/tokens and their integer indices.
    """
    def __init__(self):
        # Define special tokens
        self.pad_token = '[PAD]'
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.mask_token = '[MASK]'
        self.unk_token = '[UNK]'
        
        # Build vocabulary: a-z, special character '*', and special tokens
        self.vocab = list(string.ascii_lowercase) + ['*'] + [self.pad_token, self.sep_token, self.cls_token, self.mask_token, self.unk_token]
        
        # Create token-to-index and index-to-token mappings
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = {idx: token for idx, token in enumerate(self.vocab)}
        
        # Store vocabulary size
        self.vocab_size = len(self.vocab)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to their corresponding indices.
        """
        return [self.token2idx.get(token, self.token2idx[self.unk_token]) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert indices to their corresponding tokens.
        """
        return [self.idx2token.get(idx, self.unk_token) for idx in ids]
    
    def encode(self, sequence: List[str], max_length: int = 50, padding: bool = True, truncation: bool = True) -> np.ndarray:
        """
        Encode a sequence of tokens to their indices.
        """
        # Convert tokens to indices
        ids = self.convert_tokens_to_ids(sequence)
        
        # Truncate sequence if needed
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        
        # Pad sequence if needed
        if padding and len(ids) < max_length:
            ids = ids + [self.token2idx[self.pad_token]] * (max_length - len(ids))
        
        return np.array(ids)
    
    def decode(self, ids: List[int]) -> List[str]:
        """
        Decode a sequence of indices to their corresponding tokens.
        """
        # Convert torch tensor or numpy array to list if needed
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()
        # Now check if input is a list
        if not isinstance(ids, list):
            raise TypeError("Input must be a list of integers, got {}".format(type(ids)))
        
        return self.convert_ids_to_tokens(ids)


def sample_batch(
    words_type: str,
    tokenizer: HangmanTokenizer,
    batch_size: int = 32,
    max_seq_length: int = 50
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of game states and convert them to model inputs.
    
    Args:
        words_type: Type of words to use ('train', 'val', or 'test').
        tokenizer: Tokenizer for encoding/decoding.
        batch_size: Number of samples to generate.
        max_seq_length: Maximum sequence length.
        
    Returns:
        Tuple of (input_ids, remaining_letter_masks) as torch tensors.
    """
    # Select the appropriate word list
    if words_type == 'train':
        words = train_words
        hard_pool = hard_words_train
    elif words_type == 'val':
        words = val_words
        hard_pool = hard_words_val
    elif words_type == 'test':
        words = test_words
        hard_pool = hard_words_test
    else:
        raise ValueError(f"Invalid words_type: {words_type}. Must be 'train', 'val', or 'test'.")
    
    # Sample game states
    hard_sample_size = batch_size // 2
    normal_sample_size = batch_size - hard_sample_size
    hard_samples = sample_hard_game_states(hard_pool, hard_sample_size)
    normal_samples = sample_game_states(words, normal_sample_size)
    samples = hard_samples + normal_samples
    random.shuffle(samples)
    # samples = sample_game_states(words, batch_size)
    # Prepare inputs and outputs
    input_ids_list = []
    target_masks_list = []
    
    for sample in samples:
        # Prepare model input
        sequence = prepare_model_input(
            sample['state'], 
            sample['guessed_letters'], 
            sample['wrong_attempts'],
            max_seq_length
        )
        
        # Encode sequence
        input_ids = tokenizer.encode(sequence, max_seq_length)
        input_ids_list.append(input_ids)
        
        # Create target mask
        # 1 for letters still in the word (target), 0 for others
        target_mask = np.zeros(26)  # 26 letters in the alphabet
        for letter in sample['remaining_letters']:
            idx = ord(letter) - ord('a')
            if 0 <= idx < 26:
                target_mask[idx] = 1
        
        target_masks_list.append(target_mask)
    
    # Convert to torch tensors
    input_ids_tensor = torch.tensor(np.stack(input_ids_list), dtype=torch.long)
    target_masks_tensor = torch.tensor(np.stack(target_masks_list), dtype=torch.float)
    
    return input_ids_tensor, target_masks_tensor 