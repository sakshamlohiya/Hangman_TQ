"""
Utility functions for Hangman letter prediction model.
"""
import random
import string
import numpy as np
from typing import List, Tuple, Set, Dict


from transformers import AutoTokenizer, AutoModel

def load_tokenizer(model_name: str = "prajjwal1/bert-small"):
    """
    Load the tokenizer and model for letter prediction.
    
    Args:
        model_name: Name of the pretrained model to load.
        
    Returns:
        Tuple of (tokenizer, model)
    """
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prune the tokenizer to only keep a-z, '*', and special tokens
    vocab = tokenizer.get_vocab()
    
    # Create a new vocabulary with only the tokens we want to keep
    new_vocab = {}
    
    # Keep special tokens
    for token, idx in vocab.items():
        if token in tokenizer.all_special_tokens or token in tokenizer.all_special_tokens_extended:
            new_vocab[token] = idx
    
    # Keep a-z tokens and '*'
    for char in string.ascii_lowercase + '*':
        if char in vocab:
            new_vocab[char] = vocab[char]
    
    # Update the tokenizer's vocabulary
    tokenizer.vocab = new_vocab
    tokenizer.ids_to_tokens = {v: k for k, v in new_vocab.items()}
    
    return tokenizer



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
    return words


def generate_game_state(word: str, guessed_letters: Set[str]) -> List[str]:
    """
    Generate the current word state based on the word and guessed letters.
    
    Args:
        word: The target word.
        guessed_letters: Set of already guessed letters.
        
    Returns:
        List of characters representing word state with revealed letters and '*' for unknowns.
    """
    state = []
    for char in word:
        if char in guessed_letters:
            state.append(char)
        else:
            state.append('*')
    return state


def sample_game_states(words: List[str], num_samples: int) -> List[Dict]:
    """
    Sample realistic Hangman game states focusing on unknown beginning and ending letters.
    
    Args:
        words: List of words to sample from.
        num_samples: Number of game states to sample.
        
    Returns:
        List of realistic game states with word, state, guessed letters, and remaining letters.
    """
    samples = []
    common_vowels = set(['a', 'e', 'i', 'o', 'u'])
    common_consonants = set(['t', 'n', 's', 'r', 'h', 'l', 'd', 'c', 'm', 'f'])
    
    while len(samples) < num_samples:
        word = random.choice(words)
        
        # Bias toward mid-game states (2-4 wrong guesses is most common)
        wrong_attempts = random.choices(
            [0, 1, 2, 3, 4, 5], 
            weights=[0.05, 0.15, 0.35, 0.25, 0.15, 0.05],  # Peak at 2-3 wrong guesses
            k=1
        )[0]
        
        all_letters = set(string.ascii_lowercase)
        word_letters = set(word)
        non_word_letters = all_letters - word_letters
        
        # Identify beginning and ending regions (important for gameplay)
        word_length = len(word)
        beginning_size = min(2, word_length)
        ending_size = min(2, word_length)
        
        beginning_letters = set(word[:beginning_size])
        ending_letters = set(word[-ending_size:])
        middle_letters = word_letters - beginning_letters - ending_letters
        
        # In real games, common vowels and consonants are guessed first
        word_vowels = word_letters.intersection(common_vowels)
        word_common_consonants = word_letters.intersection(common_consonants)
        word_uncommon_letters = word_letters - word_vowels - word_common_consonants
        
        # Realistic guessing probabilities
        vowel_chance = 0.7  # 70% of vowels are guessed
        common_consonant_chance = 0.5  # 50% of common consonants are guessed
        uncommon_letter_chance = 0.2  # 20% of uncommon letters are guessed
        
        # Beginning and ending letters have lower probabilities of being guessed
        beginning_modifier = 0.7  # 30% less likely to be guessed
        ending_modifier = 0.8  # 20% less likely to be guessed
        
        guessed_letters = set()
        
        # Decide which letters are guessed based on category and position
        for letter in word_letters:
            base_probability = 0
            position_modifier = 1.0
            
            # Determine base probability by letter frequency
            if letter in word_vowels:
                base_probability = vowel_chance
            elif letter in word_common_consonants:
                base_probability = common_consonant_chance
            else:
                base_probability = uncommon_letter_chance
                
            # Apply position modifiers
            if letter in beginning_letters:
                position_modifier = beginning_modifier
            elif letter in ending_letters:
                position_modifier = ending_modifier
                
            # Final probability for this letter
            final_probability = base_probability * position_modifier
            
            # Decide if this letter is guessed
            if random.random() < final_probability:
                guessed_letters.add(letter)
        
        # Add wrong guesses from non-word letters based on wrong_attempts
        available_wrong = list(non_word_letters - guessed_letters)
        # Prefer common wrong guesses
        common_wrong = [l for l in available_wrong if l in common_vowels or l in common_consonants]
        wrong_guesses = set()
        
        # First use common letters for wrong guesses, then uncommon if needed
        if wrong_attempts <= len(common_wrong):
            wrong_guesses = set(random.sample(common_wrong, wrong_attempts))
        else:
            wrong_guesses = set(common_wrong)  # All common letters
            remaining_wrong = wrong_attempts - len(common_wrong)
            uncommon_wrong = [l for l in available_wrong if l not in common_wrong]
            if remaining_wrong > 0 and uncommon_wrong:
                wrong_guesses.update(random.sample(uncommon_wrong, min(remaining_wrong, len(uncommon_wrong))))
        
        # Combine all guessed letters
        all_guessed = guessed_letters.union(wrong_guesses)
        guessed_letters_list = list(all_guessed)
        random.shuffle(guessed_letters_list)
        
        # Generate current word state
        state = generate_game_state(word, all_guessed)
        
        # Determine the remaining unguessed letters in the word
        remaining_letters = set(word) - all_guessed
        
        # Check that at least one beginning OR ending letter is unknown
        beginning_unknown = '*' in state[:beginning_size]
        ending_unknown = '*' in state[-ending_size:]
        
        if beginning_unknown or ending_unknown:
            samples.append({
                'word': word,
                'state': state,
                'guessed_letters': guessed_letters_list,
                'wrong_attempts': wrong_attempts,
                'remaining_letters': remaining_letters
            })
    
    return samples

import random
import string
from collections import Counter
from typing import List, Dict, Tuple, Set

# --------------------------------------------------------------------------------------
# 1.  CONFIGURATION CONSTANTS
# --------------------------------------------------------------------------------------

VOWELS                 = set("aeiou")
COMMON_CONSONANTS      = set("tnsrhldcmf")          # frequent in English words
RARE_LETTERS           = set("qjxzvkwy")            # letters humans guess late
MAX_WRONG_GUESSES      = 6                          # standard Hangman limit
DIFFICULT_LENGTH_RANGE = range(3, 9)                # 3- to 8-letter words
EARLY_VOWEL_SEQUENCE   = list("eaiou")              # human-like opening moves
EARLY_CONSONANT_SEQ    = list("tnshrldc")           # common consonants after vowels
RANDOM_SEED            = 42                         # reproducibility (optional)

random.seed(RANDOM_SEED)


# --------------------------------------------------------------------------------------
# 2.  DIFFICULTY HEURISTICS
# --------------------------------------------------------------------------------------

def word_difficulty(word: str) -> float:
    """
    Heuristic difficulty score (higher == harder).
    Combines rare letters, low vowel count, and low unique-letter ratio.
    """
    letters = set(word)
    # Rarity boost
    rarity_score = 3 * len(letters & RARE_LETTERS)
    # Penalise vowels: fewer vowels -> harder
    vowel_score = (5 - len(letters & VOWELS)) * 1.5
    # Fewer unique letters relative to length -> harder (due to repetition)
    unique_ratio = len(letters) / len(word)
    repetition_score = (1 - unique_ratio) * 4
    return rarity_score + vowel_score + repetition_score


def build_hard_word_pool(words: List[str]) -> List[Tuple[str, float]]:
    """
    Filter and rank words by difficulty.
    Returns list of (word, difficulty_score), hardest first.
    """
    candidates = [
        (w, word_difficulty(w))
        for w in words
        if w.isalpha()
        and w == w.lower()
        and len(w) in DIFFICULT_LENGTH_RANGE
    ]
    # Keep top quartile hardest words
    candidates.sort(key=lambda x: x[1], reverse=True)
    cutoff = max(1, len(candidates) // 4)
    return candidates[:cutoff]


# --------------------------------------------------------------------------------------
# 3.  GAME STATE UTILITIES
# --------------------------------------------------------------------------------------

def reveal_pattern(word: str, guessed: Set[str]) -> str:
    """Return pattern string like '_ a _ _' (spaces optional)."""
    return "".join(ch if ch in guessed else "*" for ch in word)


def choose_next_guess(
    pattern: str,
    guessed: Set[str],
    candidate_words: Set[str],
    human_bias_prob: float = 0.3,
) -> str:
    """
    Very lightweight hybrid between optimal and human-biased guessing.

    * With probability `human_bias_prob`, follow the canonical human order:
      vowels first, then common consonants.
    * Otherwise, choose the letter that appears most often in the remaining
      candidate words (information-theoretic best guess).

    Returns the chosen letter (not yet in `guessed`).
    """
    remaining_letters = set(string.ascii_lowercase) - guessed

    # --- human-style bias path -------------------------------------------------
    if random.random() < human_bias_prob:
        for seq in (EARLY_VOWEL_SEQUENCE, EARLY_CONSONANT_SEQ):
            for l in seq:
                if l in remaining_letters:
                    return l
        # fallback to random rare letter
        rare_remaining = list(RARE_LETTERS & remaining_letters)
        if rare_remaining:
            return random.choice(rare_remaining)

    # --- information-gain path (frequency among candidates) --------------------
    letter_counter = Counter()
    for w in candidate_words:
        for l in set(w) & remaining_letters:
            letter_counter[l] += 1
    if not letter_counter:
        # Nothing informative left – just pick a random remaining letter
        return random.choice(sorted(remaining_letters))
    # Pick the letter that appears in most candidates (ties broken randomly)
    max_freq = max(letter_counter.values())
    best_letters = [l for l, c in letter_counter.items() if c == max_freq]
    return random.choice(best_letters)


# --------------------------------------------------------------------------------------
# 4.  SIMULATE A SINGLE GAME AND EXTRACT “HARD” STATES
# --------------------------------------------------------------------------------------

def simulate_game_states(word: str, lexicon: List[str]) -> List[Dict]:
    """
    Simulate a Hangman round against `word`, returning *selected* hard states.
    Each state dict contains:
        pattern, guessed_letters (list in guess order), wrong_attempts, next_letter
    """
    candidate_words = {w for w in lexicon if len(w) == len(word)}
    guessed: Set[str] = set()
    wrong_attempts = 0
    history = []

    while wrong_attempts < MAX_WRONG_GUESSES and "*" in reveal_pattern(word, guessed):
        pattern = reveal_pattern(word, guessed)
        next_guess = choose_next_guess(pattern, guessed, candidate_words)
        guessed.add(next_guess)

        if next_guess not in word:
            wrong_attempts += 1
        else:
            # Narrow candidate set to those matching new pattern
            pattern = reveal_pattern(word, guessed)
            candidate_words = {
                w
                for w in candidate_words
                if all(
                    (p == "*" or p == w[i]) for i, p in enumerate(pattern)
                )
                and not any((g in w) for g in guessed if g not in pattern)
            }

        # Record only *challenging* intermediate states:
        #  • at least 2 wrong guesses so far
        #  • there are still blanks at start or end
        pattern_now = reveal_pattern(word, guessed)
        if (
            wrong_attempts >= 2
            and (pattern_now[0] == "*" or pattern_now[-1] == "*")
            and "*" in pattern_now
        ):
            history.append(
                {
                    "word": word,
                    "state": pattern_now,
                    "guessed_letters": list(guessed),
                    "wrong_attempts": wrong_attempts,
                    "remaining_letters": set(word) - guessed,
                    "next_letter": None,  # placeholder, filled after break
                }
            )

    # After loop, if game not lost, we can set the “next_letter” for the last stored state
    if history and "*" in reveal_pattern(word, guessed) and wrong_attempts < MAX_WRONG_GUESSES:
        last_state = history[-1]
        # Pick the *correct* next letter (one unrevealed in word)
        last_state["next_letter"] = random.choice(list(set(word) - set(last_state["guessed_letters"])))
    return history


# --------------------------------------------------------------------------------------
# 5.  TOP-LEVEL SAMPLER
# --------------------------------------------------------------------------------------

def sample_hard_game_states(hard_pool: List[Tuple[str, float]], num_samples: int) -> List[Dict]:
    """
    Sample `num_samples` realistic *hard-mode* Hangman states.

    Returns: list of dictionaries ready for model training:
        {
          'word', 'state', 'guessed_letters', 'wrong_attempts',
          'remaining_letters', 'next_letter'
        }
    """

    # Prepare a separate lexicon for pattern matching (all lowercase alpha words)
    lexicon = [w for w, _ in hard_pool]

    # Weight selection by difficulty score^2 so very tough words are favoured
    words_only, diffs = zip(*hard_pool)
    weights = [d ** 2 for d in diffs]

    samples: List[Dict] = []
    while len(samples) < num_samples:
        word = random.choices(words_only, weights=weights, k=1)[0]
        new_states = simulate_game_states(word, lexicon)
        for st in new_states:
            samples.append(st)
            if len(samples) >= num_samples:
                break

    random.shuffle(samples)
    return samples[:num_samples]  # trim in case of slight over-collection


# --------------------------------------------------------------------------------------
# 6.  QUICK DEMO (remove or comment out in production)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal word list for illustration.  Replace with your full list.
    demo_words = [
        "jazz", "quiz", "rhythm", "babe", "fjord", "syzygy",
        "banana", "kiosk", "fluff", "lynx", "queue", "crypt"
    ]
    sliced = sample_hard_game_states(demo_words, 5)
    from pprint import pprint
    pprint(sliced)

def prepare_model_input(state: str, guessed_letters: List[str], wrong_attempts: int, max_seq_length: int = 50) -> List:
    """
    Prepare model input as a sequence: [guessed_letters, SEP, state, wrong_attempts].
    
    Args:
        state: Current word state with revealed letters and '*' for unknowns.
        guessed_letters: List of already guessed letters.
        wrong_attempts: Number of wrong attempts.
        max_seq_length: Maximum sequence length for padding.
        
    Returns:
        List of characters as input sequence.
    """
    # Special tokens
    SEP_TOKEN = '[SEP]'
    PAD_TOKEN = '[PAD]'
    
    # Create the sequence: guessed_letters + SEP + state
    sequence = guessed_letters + [SEP_TOKEN] + list(state)
    
    # Pad or truncate the sequence to max_seq_length
    if len(sequence) > max_seq_length:
        sequence = sequence[:max_seq_length]
    else:
        sequence = sequence + [PAD_TOKEN] * (max_seq_length - len(sequence))
    
    return sequence


def evaluate_guess(word: str, guess: str, guessed_letters: Set[str], wrong_attempts: int) -> Tuple[bool, Set[str], str, int]:
    """
    Evaluate a guess in the Hangman game.
    
    Args:
        word: The target word.
        guess: The guessed letter.
        guessed_letters: Set of already guessed letters.
        wrong_attempts: Current number of wrong attempts.
        
    Returns:
        Tuple of (is_correct, updated_guessed_letters, new_state, updated_wrong_attempts)
    """
    updated_guessed = guessed_letters.union({guess})
    is_correct = guess in word
    
    # Update wrong attempts if the guess is incorrect
    updated_wrong_attempts = wrong_attempts
    if not is_correct:
        updated_wrong_attempts += 1
    
    new_state = generate_game_state(word, updated_guessed)
    return is_correct, updated_guessed, new_state, updated_wrong_attempts


def is_game_won(state: str) -> bool:
    """
    Check if the game is won.
    
    Args:
        state: Current word state with revealed letters and '*' for unknowns.
        
    Returns:
        True if the game is won, False otherwise.
    """
    return '*' not in state


def is_game_lost(wrong_attempts: int, max_attempts: int = 6) -> bool:
    """
    Check if the game is lost.
    
    Args:
        wrong_attempts: Current number of wrong attempts.
        max_attempts: Maximum number of allowed wrong attempts.
        
    Returns:
        True if the game is lost, False otherwise.
    """
    return wrong_attempts >= max_attempts