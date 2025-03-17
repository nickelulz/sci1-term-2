import sys
import numpy as np
import random
from itertools import product
sys.path.insert(1, '../src/')  # Ensure custom modules are accessible

from coins import CoinFlipEngine
from reverse_engineer import Calix
from generator import generate_possible_combinations
from flips import perform_coin_flips

def generate_markov_rule(size: int, memory_depth: int) -> dict:
    """Generates a single Markov rule (transition map) for one coin."""
    combinations = list(product(range(size), repeat=memory_depth))
    return {comb: random.randint(0, size - 1) for comb in combinations}

def markov_reaches_all(markov_system: list, size: int) -> bool:
    """Checks if a Markov system can reach all states."""
    reachable_states = set()
    
    for markov in markov_system:
        reachable_states.update(markov.values())
    
    return len(reachable_states) == size

def generate_random_markov_system(size: int, memory_depth: int, num_samples=5000, debug=False):
    """
    Generates valid Markov systems where each system consists of `size` distinct
    Markov rules (one per state) and can reach all states.
    """
    if debug:
        print(f'Generating {num_samples} full Markov systems...')
    
    valid_markov_systems = []
    attempts = 0
    
    while len(valid_markov_systems) < num_samples:
        # Step 1: Generate `size` distinct Markov rules
        markov_system = [generate_markov_rule(size, memory_depth) for _ in range(size)]
        
        # Step 2: Check if the system reaches all states
        if markov_reaches_all(markov_system, size):
            valid_markov_systems.append(markov_system)
        
        attempts += 1
        if attempts % 1000 == 0 and debug:
            print(f'Attempted {attempts} generations, valid: {len(valid_markov_systems)}')
    
    if debug:
        print(f'Generated {num_samples} valid Markov systems.')
    return valid_markov_systems

# --- Error Metric ---
def test_model(guess_model, test_data):
    """Compares predicted states to actual test states."""
    predictions = []
    for _ in range(10):
        predictions.append(perform_coin_flips(guess_model, len(test_data)).flip_history)
        guess_model.reset_markov()
    actual = test_data

    accuracies = [np.mean(prediction == actual) for prediction in predictions]

    """
    pred_dist = np.bincount(prediction, minlength=2) / len(prediction)
    act_dist = np.bincount(actual, minlength=2) / len(actual)

    if np.shape(pred_dist) != np.shape(act_dist):
        # print(np.shape(pred_dist), np.shape(act_dist))
        if len(act_dist) < len(act_dist):
            pred_dist = np.pad(pred_dist, (0, len(act_dist) - len(pred_dist)), mode='constant')
        else:
            act_dist = np.pad(act_dist, (0, len(pred_dist) - len(act_dist)), mode='constant') 

    # calculate accuracy as the euclidean distance between the two distributions
    euclid_dist = np.linalg.norm(pred_dist - act_dist)
    """

    return np.mean(accuracies)

# --- Ajani Reverse Engineering ---
def Ajani(train_data, test_data, size, memory_depth, num_samples=1, debug=False):
    """
    Reverse engineers the best Markov model by sampling different full Markov systems.
    """
    sampled_markov_systems = generate_random_markov_system(size, memory_depth, num_samples)
    num_markovs = len(sampled_markov_systems)

    best_guess = None
    best_guess_accuracy = 0

    for i, markov_system in enumerate(sampled_markov_systems):
        if debug:
            print(f'Estimating - {i+1}/{num_markovs}')
        guess = Calix(train_data, 
                      markov_system, memory_depth, size, 
                      delta=0.01, debug=False, benchmark_flips=0)
        accuracy = test_model(guess, test_data)

        if accuracy > best_guess_accuracy:
            best_guess = guess
            best_guess_accuracy = accuracy

    return best_guess, best_guess_accuracy
