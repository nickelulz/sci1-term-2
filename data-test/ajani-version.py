import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.insert(1, '../src/')
from coins import CoinFlipEngine
from reverse_engineer import Calix
from generator import generate_all_possible_markov_system

def test_model(guess_model, test_data):
    raise NotImplementedError()

def Ajani(train_data, test_data, size, memory_depth):
    """
    we have to manually readapt ajani b/c the original version calculates
    error based on memory depth :(
    """
    print('Generating All Markovs')
    all_possible_markovs = generate_all_possible_markov_system(size, memory_depth)
    num_markovs = len(all_possible_markovs)

    print(f'Generated {num_markovs} markovs.')

    best_guess = None
    best_guess_error = np.inf

    print('Beginning estimation..')
    for i, markov in enumerate(all_possible_markovs):
        if debug:
            print(f'Ajani Estimation {i+1}/{num_markovs}')
        guess = Calix(train_data, markov, memory_depth, 
                      size, delta=0.01, debug=False, 
                      benchmark_flips=0) # do not benchmark
        error = test_model(guess, test_data)

        if error < best_guess_error:
            best_guess = guess
            best_guess_error = error

    return best_guess, best_guess_error

# Function to load weather categories from a file
def load_weather_data(filepath):
    """Reads newline-separated weather categories and returns a NumPy array."""
    with open(filepath, "r") as f:
        return np.array([int(line.strip()) for line in f if line.strip()]).reshape(-1, 1)

def split_data(data):
    split_idx = int(len(data) * 0.75)
    return data[:split_idx], data[split_idx:]

"""
For the sake of simplicity, we're going to follow the "Bi-Gram Markov Model" (no clue what that means),
i.e.: these have a memory depth of 2.
"""
MEMORY_DEPTH = 1

def main():
    # Define weather state labels
    weather_labels = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Rain", "Heavy Rain"]

    # Load UBC and Richmond data
    ubc_data = load_weather_data("data/ubc-train.csv")
    richmond_data = load_weather_data("data/richmond-test.csv")

    # Split UBC and Richmond data
    ubc_train, ubc_test = split_data(ubc_data)
    richmond_train, richmond_test = split_data(richmond_data)

    n_states = 6
    model, error = Ajani(ubc_train, ubc_test, size=n_states, memory_depth=MEMORY_DEPTH)
    print(model)
    print('error: ', error)

if __name__ == '__main__':
    main()
