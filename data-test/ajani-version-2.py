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

def generate_random_markov_system(size: int, memory_depth: int, num_samples=5000):
    """
    Generates valid Markov systems where each system consists of `size` distinct
    Markov rules (one per state) and can reach all states.
    """
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
        if attempts % 1000 == 0:
            print(f'Attempted {attempts} generations, valid: {len(valid_markov_systems)}')
    
    print(f'Generated {num_samples} valid Markov systems.')
    return valid_markov_systems

# --- Error Metric ---
def test_model(guess_model, test_data):
    """Compares predicted states to actual test states."""
    prediction = perform_coin_flips(guess_model, len(test_data)).flip_history
    actual = test_data
    accuracy = np.mean(prediction == actual)
    return 1.0 - accuracy

# --- Ajani Reverse Engineering ---
def Ajani(train_data, test_data, size, memory_depth, num_samples=1):
    """
    Reverse engineers the best Markov model by sampling different full Markov systems.
    """
    sampled_markov_systems = generate_random_markov_system(size, memory_depth, num_samples)
    num_markovs = len(sampled_markov_systems)

    best_guess = None
    best_guess_error = np.inf

    for i, markov_system in enumerate(sampled_markov_systems):
        print(f'Estimating - {i+1}/{num_markovs}')
        guess = Calix(train_data.flatten().tolist(), 
                      markov_system, memory_depth, size, 
                      delta=0.01, debug=False, benchmark_flips=0)
        error = test_model(guess, test_data.flatten().tolist())

        if error < best_guess_error:
            best_guess = guess
            best_guess_error = error

    return best_guess, best_guess_error

# --- Load Weather Data ---
def load_weather_data(filepath):
    """Reads newline-separated weather categories and returns a NumPy array."""
    with open(filepath, "r") as f:
        return np.array([int(line.strip()) for line in f if line.strip()]).reshape(-1, 1)

# --- Split Data ---
def split_data(data):
    split_idx = int(len(data) * 0.75)
    return data[:split_idx], data[split_idx:]

# --- Constants ---
MEMORY_DEPTH = 2
N_STATES = 6
NUM_SAMPLES = 50  # Adjustable for tradeoff between speed vs accuracy

def main():
    # Load UBC and Richmond data
    ubc_data = load_weather_data("data/ubc-train.csv")
    richmond_data = load_weather_data("data/richmond-test.csv")

    # Split datasets
    ubc_train, ubc_test = split_data(ubc_data)
    richmond_train, richmond_test = split_data(richmond_data)

    # Train & Evaluate
    print("[UBC Model]")
    ubc_model, ubc_error = Ajani(ubc_train, ubc_test, N_STATES, MEMORY_DEPTH, NUM_SAMPLES)
    print("UBC Model Error:", ubc_error)

    # print("\n[Richmond Model]")
    # richmond_model, richmond_error = Ajani(richmond_train, richmond_test, N_STATES, MEMORY_DEPTH, NUM_SAMPLES)
    # print("Richmond Model Error:", richmond_error)

if __name__ == '__main__':
    main()

