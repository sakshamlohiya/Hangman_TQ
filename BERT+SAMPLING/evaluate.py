"""
Evaluation script for Hangman letter prediction model.
"""
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import string

from dataloader import HangmanTokenizer, create_data_loaders
from model import create_model
from utils import load_words, generate_game_state, evaluate_guess, is_game_won, is_game_lost, sample_game_states


def setup_logging(log_dir: str) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'evaluation.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_model(checkpoint_path: str, model_type: str, config: Dict, device: torch.device) -> nn.Module:
    """
    Load a model from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint.
        model_type: Type of model.
        config: Model configuration.
        device: Device to load model on.
        
    Returns:
        Loaded model.
    """
    model = create_model(model_type, config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logging.info(f"Loaded model from {checkpoint_path}")
    
    return model


def evaluate_model(model: nn.Module, test_loader: torch.utils.data.DataLoader, 
                 device: torch.device) -> Dict:
    """
    Evaluate model on test data.
    
    Args:
        model: Model to evaluate.
        test_loader: DataLoader for test data.
        device: Device to evaluate on.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            # Move tensors to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Convert outputs to probabilities if needed
            if isinstance(outputs, dict):
                predictions = outputs['probs']
            else:
                predictions = outputs
            
            # Collect predictions and targets
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Convert to letter indices
    pred_indices = np.argmax(all_predictions, axis=1)
    
    # For multi-label targets, convert to indices of highest probability
    target_indices = np.argmax(all_targets, axis=1)
    
    # Compute accuracy
    accuracy = np.mean(pred_indices == target_indices)
    
    # Compute top-k accuracy
    k_values = [1, 3, 5]
    top_k_accuracy = {}
    
    for k in k_values:
        top_k_preds = np.argsort(all_predictions, axis=1)[:, -k:]
        top_k_correct = 0
        
        for i, target_idx in enumerate(target_indices):
            if target_idx in top_k_preds[i]:
                top_k_correct += 1
        
        top_k_accuracy[f'top_{k}_accuracy'] = top_k_correct / len(target_indices)
    
    # Generate confusion matrix
    cm = confusion_matrix(target_indices, pred_indices)
    
    # Generate classification report
    letters = list(string.ascii_lowercase)
    report = classification_report(target_indices, pred_indices, 
                                  target_names=letters, 
                                  output_dict=True)
    
    # Calculate average number of steps to solution
    # This would be better to compute in a separate function with actual gameplay simulation
    
    return {
        'accuracy': accuracy,
        'top_k_accuracy': top_k_accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }


def simulate_hangman_games(model: nn.Module, words: List[str], tokenizer: HangmanTokenizer, 
                         device: torch.device, num_games: int = 1000, max_attempts: int = 6) -> Dict:
    """
    Simulate Hangman games to evaluate model performance.
    
    Args:
        model: Trained model.
        words: List of words for simulation.
        tokenizer: Tokenizer for encoding inputs.
        device: Device to run model on.
        num_games: Number of games to simulate.
        max_attempts: Maximum number of allowed wrong attempts.
        
    Returns:
        Dictionary of simulation metrics.
    """
    model.eval()
    
    # Randomly sample words for simulation
    sampled_words = random.sample(words, min(num_games, len(words)))
    
    wins = 0
    losses = 0
    total_attempts = 0
    attempt_distribution = [0] * (max_attempts + 1)  # +1 for successful games
    
    for word in tqdm(sampled_words, desc="Simulating games"):
        # Initialize game state
        guessed_letters = set()
        wrong_attempts = 0
        state = generate_game_state(word, guessed_letters)
        
        while not is_game_won(state) and not is_game_lost(wrong_attempts, max_attempts):
            # Prepare model input
            # First, convert guessed letters to a sorted list
            guessed_list = sorted(list(guessed_letters))
            
            # Create input sequence: [guessed_letters, SEP, state, wrong_attempts]
            sequence = guessed_list + ['[SEP]'] + list(state) + [str(wrong_attempts)]
            
            # Encode sequence
            input_ids = tokenizer.encode(sequence)
            
            # Move to device
            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                
                # Convert outputs to probabilities if needed
                if isinstance(outputs, dict):
                    predictions = outputs['probs']
                else:
                    predictions = outputs
            
            # Get top letter predictions
            probs = predictions.squeeze().cpu().numpy()
            
            # Filter out already guessed letters
            for i, letter in enumerate(string.ascii_lowercase):
                if letter in guessed_letters:
                    probs[i] = 0
            
            # Get the letter with highest probability
            next_letter = string.ascii_lowercase[np.argmax(probs)]
            
            # Evaluate guess
            is_correct, guessed_letters, state, wrong_attempts = evaluate_guess(
                word, next_letter, guessed_letters, wrong_attempts
            )
            
            # Update total attempts
            total_attempts += 1
        
        # Check game outcome
        if is_game_won(state):
            wins += 1
            attempt_distribution[wrong_attempts] += 1  # Successful games classified by wrong attempts
        else:
            losses += 1
            attempt_distribution[max_attempts] += 1  # All unsuccessful games in the last bin
    
    # Calculate metrics
    win_rate = wins / len(sampled_words)
    avg_attempts = total_attempts / len(sampled_words)
    
    return {
        'num_games': len(sampled_words),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_attempts': avg_attempts,
        'attempt_distribution': attempt_distribution
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix.
        save_path: Path to save the plot.
    """
    plt.figure(figsize=(12, 10))
    classes = list(string.ascii_lowercase)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Letter')
    plt.xlabel('Predicted Letter')
    plt.savefig(save_path)
    plt.close()


def plot_top_k_accuracy(top_k_accuracy: Dict, save_path: str) -> None:
    """
    Plot top-k accuracy.
    
    Args:
        top_k_accuracy: Dictionary of top-k accuracy values.
        save_path: Path to save the plot.
    """
    k_values = [int(k.split('_')[1]) for k in top_k_accuracy.keys()]
    accuracies = list(top_k_accuracy.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(k_values, accuracies)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Top-k Accuracy')
    plt.ylim([0, 1])
    for i, acc in enumerate(accuracies):
        plt.text(k_values[i], acc + 0.02, f'{acc:.3f}', ha='center')
    plt.savefig(save_path)
    plt.close()


def plot_attempt_distribution(attempt_distribution: List[int], save_path: str, max_attempts: int = 6) -> None:
    """
    Plot attempt distribution.
    
    Args:
        attempt_distribution: List of attempt counts.
        save_path: Path to save the plot.
        max_attempts: Maximum number of allowed wrong attempts.
    """
    plt.figure(figsize=(10, 6))
    labels = [str(i) for i in range(max_attempts)] + [f'{max_attempts}+']
    plt.bar(labels, attempt_distribution)
    plt.xlabel('Wrong Attempts')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Wrong Attempts in Hangman Games')
    for i, count in enumerate(attempt_distribution):
        plt.text(i, count + 5, str(count), ha='center')
    plt.savefig(save_path)
    plt.close()


def evaluate(args: argparse.Namespace) -> None:
    """
    Evaluate the model.
    
    Args:
        args: Command-line arguments.
    """
    # Set up logging
    setup_logging(args.log_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create tokenizer
    tokenizer = HangmanTokenizer()
    
    # Create model configuration
    model_config = {
        'vocab_size': tokenizer.vocab_size,
        'hidden_size': args.hidden_size,
        'num_hidden_layers': args.num_hidden_layers,
        'num_attention_heads': args.num_attention_heads,
    }
    
    # Load model
    model = load_model(args.checkpoint_path, args.model_type, model_config, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate on test set
    if args.eval_test_set:
        logging.info("Evaluating on test set")
        
        # Create data loaders
        _, _, test_loader = create_data_loaders(
            args.train_file,
            args.val_file,
            args.test_file,
            tokenizer,
            args.train_samples,
            args.val_samples,
            args.test_samples,
            args.batch_size,
            args.max_seq_length,
            args.cache_dir,
            args.num_workers
        )
        
        # Evaluate model
        test_metrics = evaluate_model(model, test_loader, device)
        
        # Log metrics
        logging.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        for k, acc in test_metrics['top_k_accuracy'].items():
            logging.info(f"Test {k}: {acc:.4f}")
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'accuracy': test_metrics['accuracy'],
                'top_k_accuracy': test_metrics['top_k_accuracy'],
                'classification_report': test_metrics['classification_report']
            }, f, indent=2)
        
        # Plot confusion matrix
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(test_metrics['confusion_matrix'], cm_path)
        
        # Plot top-k accuracy
        top_k_path = os.path.join(args.output_dir, 'top_k_accuracy.png')
        plot_top_k_accuracy(test_metrics['top_k_accuracy'], top_k_path)
    
    # Simulate Hangman games
    if args.simulate_games:
        logging.info(f"Simulating {args.num_games} Hangman games")
        
        # Load test words
        test_words = load_words(args.test_file)
        
        # Simulate games
        game_metrics = simulate_hangman_games(
            model, 
            test_words, 
            tokenizer, 
            device, 
            args.num_games, 
            args.max_attempts
        )
        
        # Log metrics
        logging.info(f"Win rate: {game_metrics['win_rate']:.4f}")
        logging.info(f"Average attempts: {game_metrics['avg_attempts']:.2f}")
        
        # Save metrics
        game_metrics_path = os.path.join(args.output_dir, 'game_metrics.json')
        with open(game_metrics_path, 'w') as f:
            json.dump(game_metrics, f, indent=2)
        
        # Plot attempt distribution
        attempt_path = os.path.join(args.output_dir, 'attempt_distribution.png')
        plot_attempt_distribution(game_metrics['attempt_distribution'], attempt_path, args.max_attempts)
    
    logging.info("Evaluation completed")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hangman letter prediction model")
    
    # Data arguments
    parser.add_argument('--train_file', type=str, default='train_words.txt',
                        help='Path to training words file')
    parser.add_argument('--val_file', type=str, default='val_words.txt',
                        help='Path to validation words file')
    parser.add_argument('--test_file', type=str, default='test_words.txt',
                        help='Path to test words file')
    parser.add_argument('--cache_dir', type=str, default='cache',
                        help='Directory to cache sampled game states')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['bert', 'transformer', 'cnn'],
                        help='Type of model to use')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size of the model')
    parser.add_argument('--num_hidden_layers', type=int, default=4,
                        help='Number of hidden layers')
    parser.add_argument('--num_attention_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--max_seq_length', type=int, default=50,
                        help='Maximum sequence length')
    
    # Evaluation arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--train_samples', type=int, default=1000,
                        help='Number of game states to sample for training')
    parser.add_argument('--val_samples', type=int, default=500,
                        help='Number of game states to sample for validation')
    parser.add_argument('--test_samples', type=int, default=1000,
                        help='Number of game states to sample for testing')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--eval_test_set', action='store_true',
                        help='Evaluate model on test set')
    parser.add_argument('--simulate_games', action='store_true',
                        help='Simulate Hangman games')
    parser.add_argument('--num_games', type=int, default=1000,
                        help='Number of games to simulate')
    parser.add_argument('--max_attempts', type=int, default=6,
                        help='Maximum number of wrong attempts allowed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == '__main__':
    main()