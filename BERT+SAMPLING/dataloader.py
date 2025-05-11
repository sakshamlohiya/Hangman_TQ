"""
Simplified DataLoader for Hangman letter prediction model.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional

from simple_dataloader import HangmanTokenizer, sample_batch


class HangmanSamplingDataset(Dataset):
    """
    A dataset that samples game states on-the-fly.
    """
    def __init__(
        self, 
        words_type: str,
        tokenizer: HangmanTokenizer,
        max_seq_length: int = 50
    ):
        """
        Initialize the dataset.
        
        Args:
            words_type: Type of words to use ('train', 'val', or 'test').
            tokenizer: Tokenizer for encoding/decoding.
            max_seq_length: Maximum sequence length.
        """
        self.words_type = words_type
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __len__(self) -> int:
        """
        Return a fixed length to satisfy DataLoader requirements.
        Since we're sampling on-the-fly, this is just a virtual size.
        """
        return 10000  # Arbitrary number, doesn't affect sampling
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a single game state.
        """
        # Sample a batch of size 1
        input_ids, target_masks = sample_batch(
            words_type=self.words_type,
            tokenizer=self.tokenizer,
            batch_size=1,
            max_seq_length=self.max_seq_length
        )
        
        # Return the single sample
        return input_ids[0], target_masks[0]


def create_datasets(
    tokenizer: HangmanTokenizer,
    max_seq_length: int = 50
) -> Tuple[HangmanSamplingDataset, HangmanSamplingDataset, HangmanSamplingDataset]:
    """
    Create datasets for training, validation, and testing.
    
    Args:
        tokenizer: Tokenizer for encoding/decoding.
        max_seq_length: Maximum sequence length.
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    # Create datasets
    train_dataset = HangmanSamplingDataset('train', tokenizer, max_seq_length)
    val_dataset = HangmanSamplingDataset('val', tokenizer, max_seq_length)
    test_dataset = HangmanSamplingDataset('test', tokenizer, max_seq_length)
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    tokenizer: HangmanTokenizer,
    batch_size: int = 32,
    max_seq_length: int = 50,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        tokenizer: Tokenizer for encoding/decoding.
        batch_size: Batch size.
        max_seq_length: Maximum sequence length.
        num_workers: Number of worker processes for data loading.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, max_seq_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class BatchSamplingLoader:
    """
    An even simpler loader that directly samples batches without using Dataset/DataLoader.
    """
    def __init__(
        self,
        words_type: str,
        tokenizer: HangmanTokenizer,
        batch_size: int = 32,
        max_seq_length: int = 50,
    ):
        """
        Initialize the batch sampling loader.
        
        Args:
            words_type: Type of words to use ('train', 'val', or 'test').
            tokenizer: Tokenizer for encoding/decoding.
            batch_size: Batch size.
            max_seq_length: Maximum sequence length.
        """
        self.words_type = words_type
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
    
    def __iter__(self):
        """
        Create an iterator that samples batches.
        """
        yield sample_batch(
            words_type=self.words_type,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length
        )


def create_simple_loaders(
    tokenizer: HangmanTokenizer,
    batch_size: int = 32,
    max_seq_length: int = 50,
) -> Tuple[BatchSamplingLoader, BatchSamplingLoader, BatchSamplingLoader]:
    """
    Create simple batch sampling loaders for training, validation, and testing.
    
    Args:
        tokenizer: Tokenizer for encoding/decoding.
        batch_size: Batch size.
        max_seq_length: Maximum sequence length.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_loader = BatchSamplingLoader(
        'train', tokenizer, batch_size, max_seq_length
    )
    
    val_loader = BatchSamplingLoader(
        'val', tokenizer, batch_size, max_seq_length
    )
    
    test_loader = BatchSamplingLoader(
        'test', tokenizer, batch_size, max_seq_length
    )
    
    return train_loader, val_loader, test_loader