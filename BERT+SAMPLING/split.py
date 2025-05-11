"""
Split the 250,000 words into train/validation/test sets while ensuring proportional representation.
"""
import argparse
import os
import random
import nltk
from nltk.corpus import words as nltk_words
from typing import List, Tuple

# Download the NLTK words corpus if not already downloaded
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')
    print("Downloaded NLTK words corpus")


def load_words(file_path: str) -> List[str]:
    """
    Load words from a text file.
    
    Args:
        file_path: Path to the text file containing words.
        
    Returns:
        List of words.
    """
    with open(file_path, 'r') as f:
        words = [line.strip() for line in f.readlines()]
    print(f"Successfully loaded {len(words)} words from file")
    return words

oxford_words_set = set(w.lower() for w in nltk_words.words())
def is_oxford_word(word: str, oxford_words_set=oxford_words_set) -> bool:
    """
    Check if a word is in the Oxford English Dictionary (approximated by NLTK's words corpus).
    
    Args:
        word: Word to check.
        oxford_words_set: Pre-computed set of Oxford words (to avoid repeated computation).
        
    Returns:
        True if the word is in the Oxford English Dictionary, False otherwise.
    """
    return word.lower() in oxford_words_set


def split_words(words: List[str], train_ratio: float = 0.9, val_ratio: float = 0.05) -> Tuple[List[str], List[str], List[str]]:
    """
    Split the words into train/validation/test sets while ensuring proportional representation.
    
    Args:
        words: List of words.
        train_ratio: Ratio of words to include in the training set.
        val_ratio: Ratio of words to include in the validation set.
        
    Returns:
        Tuple of (train_words, val_words, test_words).
    """
    # Identify Oxford and non-Oxford words
    oxford_words = []
    non_oxford_words = []
    
    print("Categorizing words as Oxford or non-Oxford...")
    from tqdm import tqdm
    for word in tqdm(words, desc="Categorizing words"):
        if is_oxford_word(word):
            oxford_words.append(word)
        else:
            non_oxford_words.append(word)
    
    print(f"Found {len(oxford_words)} Oxford words and {len(non_oxford_words)} non-Oxford words")
    
    # Shuffle the words
    random.shuffle(oxford_words)
    random.shuffle(non_oxford_words)
    print("Shuffled word lists")
    
    # Calculate split sizes for Oxford words
    oxford_train_size = int(len(oxford_words) * train_ratio)
    oxford_val_size = int(len(oxford_words) * val_ratio)
    
    # Calculate split sizes for non-Oxford words
    non_oxford_train_size = int(len(non_oxford_words) * train_ratio)
    non_oxford_val_size = int(len(non_oxford_words) * val_ratio)
    
    print(f"Oxford words split: {oxford_train_size} train, {oxford_val_size} validation, {len(oxford_words) - oxford_train_size - oxford_val_size} test")
    print(f"Non-Oxford words split: {non_oxford_train_size} train, {non_oxford_val_size} validation, {len(non_oxford_words) - non_oxford_train_size - non_oxford_val_size} test")
    
    # Split Oxford words
    oxford_train = oxford_words[:oxford_train_size]
    oxford_val = oxford_words[oxford_train_size:oxford_train_size + oxford_val_size]
    oxford_test = oxford_words[oxford_train_size + oxford_val_size:]
    
    # Split non-Oxford words
    non_oxford_train = non_oxford_words[:non_oxford_train_size]
    non_oxford_val = non_oxford_words[non_oxford_train_size:non_oxford_train_size + non_oxford_val_size]
    non_oxford_test = non_oxford_words[non_oxford_train_size + non_oxford_val_size:]
    
    # Combine Oxford and non-Oxford words
    train_words = oxford_train + non_oxford_train
    val_words = oxford_val + non_oxford_val
    test_words = oxford_test + non_oxford_test
    
    print("Combined Oxford and non-Oxford words for each split")
    
    # Shuffle again to mix Oxford and non-Oxford words
    random.shuffle(train_words)
    random.shuffle(val_words)
    random.shuffle(test_words)
    
    print("Final shuffling complete")
    print(f"Final split sizes: {len(train_words)} train, {len(val_words)} validation, {len(test_words)} test")
    
    return train_words, val_words, test_words


def save_words(words: List[str], file_path: str) -> None:
    """
    Save words to a text file.
    
    Args:
        words: List of words.
        file_path: Path to the output file.
    """
    with open(file_path, 'w') as f:
        for word in words:
            f.write(word + '\n')
    print(f"Successfully saved {len(words)} words to {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Split words into train/validation/test sets.")
    parser.add_argument('--input', type=str, default='words_250000_train.txt', help='Input file containing words')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for split files')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Ratio of words for training set')
    parser.add_argument('--val_ratio', type=float, default=0.05, help='Ratio of words for validation set')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")
    
    # Set random seed
    random.seed(args.seed)
    print(f"Random seed set to {args.seed}")
    
    # Load words
    print(f"Loading words from {args.input}...")
    words = load_words(args.input)
    print(f"Loaded {len(words)} words from {args.input}")
    
    # Split words
    print(f"Splitting words with ratios: {args.train_ratio} train, {args.val_ratio} validation, {1-args.train_ratio-args.val_ratio} test")
    train_words, val_words, test_words = split_words(words, args.train_ratio, args.val_ratio)
    
    # Create output directory if it doesn't exist
    print(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save words
    train_path = os.path.join(args.output_dir, 'train_words.txt')
    val_path = os.path.join(args.output_dir, 'val_words.txt')
    test_path = os.path.join(args.output_dir, 'test_words.txt')
    
    print("Saving split datasets...")
    save_words(train_words, train_path)
    save_words(val_words, val_path)
    save_words(test_words, test_path)
    
    print(f"Train set: {len(train_words)} words saved to {train_path}")
    print(f"Validation set: {len(val_words)} words saved to {val_path}")
    print(f"Test set: {len(test_words)} words saved to {test_path}")
    print("Data splitting complete!")


if __name__ == '__main__':
    print("Starting word splitting process...")
    main()
    print("Process completed successfully!")
    
# give running command: python split.py --input words_250000_train.txt --output_dir data --train_ratio 0.9 --val_ratio 0.05 --seed 42