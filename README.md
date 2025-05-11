Trexquant Hangman


Problem Definition:
When a user plays Hangman, the server first selects a secret word at random from a list. The server then returns a row of underscores (space separated) - one for each letter in the secret word - and asks the user to guess a letter. If the user guesses a letter that is in the word, the word is redisplayed with all instances of that letter shown in the correct positions, along with any letters correctly guessed on previous turns. If the letter does not appear in the word, the user is charged with an incorrect guess. The user keeps guessing letters until either 
The user has correctly guessed all the letters in the word  
The user has made six incorrect guesses.

Required Solution/state:
You are provided with a basic, working algorithm. This algorithm will match the provided masked string (e.g. a _ _ l e) to all possible words in the dictionary, tabulate the frequency of letters appearing in these possible words, and then guess the letter with the highest frequency of appearance that has not already been guessed. If there are no remaining words that match then it will default back to the character frequency distribution of the entire dictionary.
This benchmark strategy is successful approximately 18% of the time. Your task is to design an algorithm that significantly outperforms this benchmark.

Approaches : 
	
	Approach 1 : 
Approach 2 : Adaptive Natural Language Processing (NLP) - Ngrams + Catboost Classifier 
Approach 3 : Reinforcement Learning (RL) GRPO Based*


Hangman Strategy 01: BERT BASE Model

Overview

We've developed an AI system that can intelligently predict the next best letter to guess in a Hangman game. The system uses a deep learning model that analyzes the current game state (revealed letters, guessed letters, and wrong attempts) to make predictions that maximize the chance of winning.

Dataset Creation

Rather than using static datasets, we implemented a dynamic data generation approach:

1.⁠ ⁠Word Selection: We curated word lists from given dictionary, split into train/validation/test sets
2.⁠ ⁠Game State Simulation: 
   - Implemented realistic Hangman gameplay simulations with ⁠ sample_game_states ⁠ and ⁠ 	 sample_hard_game_states .⁠
   - Emphasized mid-game scenarios (2-4 wrong guesses) when the game is most challenging
   - Biased toward game states with unknown beginning and ending letters, which represent difficult decision points
   - Incorporated human-like guessing patterns where common vowels and consonants are guessed earlier

3.⁠ ⁠Difficult Word Selection:
   - Created a difficulty heuristic incorporating letter rarity, vowel count, and unique letter ratio
   - Developed ⁠ build_hard_word_pool ⁠ to identify challenging words for training
   - Mixed both normal and hard samples for balanced training (50/50 split)

Input Representation

For each game state, we created structured input sequences:
•⁠  ⁠Guessed letters + [SEP] token + current word state
•⁠  ⁠Current word state uses revealed letters and '*' for unknown positions
•⁠  ⁠Sequences padded/truncated to consistent length (50 tokens)

Model Architecture

We implemented several model variants, with the best performance from:

Pretrained BERT Model
•⁠  ⁠Based on "prajjwal1/bert-small" with embedding layer adjusted for our vocabulary
•⁠  ⁠Fine-tuned for the Hangman task while preserving pretrained knowledge
•⁠  ⁠Tokenizer customized to handle our specific vocabulary (a-z, '*', special tokens)

Multi-label Classification
•⁠  ⁠Output layer produces 26 logits (one per letter)
•⁠  ⁠Uses Binary Cross Entropy loss for multi-label classification
•⁠  ⁠Each output represents the probability that a letter appears in the remaining unrevealed positions

Training Strategy
•⁠  ⁠Mixed batch composition with 50% normal samples and 50% difficult samples
•⁠  ⁠Learning rate: 1e-4 with weight decay
•⁠  ⁠Periodic evaluation and checkpoint saving
•⁠  ⁠Custom metrics to evaluate letter prediction accuracy

Model Evaluation

The model was evaluated on its ability to:
1.⁠ ⁠Predict letters that actually appear in the remaining unrevealed word
3.⁠ ⁠Complete games successfully with minimal wrong guesses

Key Innovations

1.⁠ ⁠Dynamic Sampling: Instead of static datasets, we generate infinite training examples through gameplay simulation
2.⁠ ⁠Difficulty-Aware Training: Focused on challenging game states that require deeper reasoning

Conclusion

Our approach successfully combines deep learning techniques with game-specific knowledge to create an effective Hangman letter prediction system. By simulating realistic gameplay and focusing on challenging scenarios, we created a model that can provide intelligent guessing strategies in difficult game states.


Hangman Strategy 02: Adaptive NLP - Ngrams + Catboost Classifier

In building an effective Hangman solver, my focus has been on three key elements:
Character relationships within the word.
Previously guessed characters and their influence on future guesses.
The length and structure of the unknown word.
Core Idea
At each game state, the goal is to guess the next most probable letter—irrespective of its position—based on the current partial word and prior guesses. To achieve this, I compute a summed likelihood of each of the 26 letters over all blank positions in the current word pattern.
Probability Estimation
Probabilities are estimated using character-level n-gram frequencies (from unigrams up to 5-grams) extracted from a large training dictionary. For any given position in the word (with known and unknown characters), I calculate how likely each letter is to appear based on matching n-gram patterns from the dictionary. These probabilities are then aggregated across all positions for each letter to determine the best overall guess.
Adaptive Weighting Strategy
To prevent overfitting to higher-order n-grams when little context is known, I use an adaptive weighted average of the n-gram probabilities:
When fewer characters in the word are known, unigrams are weighted more heavily, as longer n-grams are unreliable in low-information contexts.
As more letters in the word are revealed, the weights gradually shift toward higher-order n-grams, which become more informative with additional context.
As number of known letter increases and we have enough information, Catboost Classifier is used for prediction of next letter
This dynamic weighting helps balance between local character frequency and longer pattern dependencies.
Additional Considerations
Previously guessed characters (both correct and incorrect) are excluded from future guesses to avoid redundancy.
The system dynamically filters the dictionary to only include candidate words consistent with the current known word pattern and guesses.
This strategy enables informed letter prediction while maintaining adaptability across different stages of the game. It balances statistical rigor with practical heuristics to maximize the solver’s accuracy and efficiency.





Approach 3 : Reinforcement Learning (RL) GRPO Based *

I tried my hand at building a reinforcement learning-enabled model to play Hangman but due to time and computational constraints during supervised learning I was not able to use it in the game .
The goal is a win percentage greater than 50%.The problem statement seemed to lend itself well towards RL as it’s a game simulation based on current state with rewards and penalty for correct and incorrect guesses. Generalized Reinforcement Policy Optimization (GRPO) is a method which uses a gradient method for policy optimization. 
GRPO fits Hangman well because it learns effective guessing strategies from limited feedback and handles discrete letter choices naturally. Its stability and ability to improve through trial and error make it a strong fit for the game's uncertain and partially revealed state.

