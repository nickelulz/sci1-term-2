import numpy as np
from coins import CoinFlipEngine
from error import calculate_model_error
from generator import generate_all_possible_markov_system

# algorithms
from re_algorithms.sleeper import Sleeper

def reverse_engineer_model(output_data_sequence: list[int], input_model: CoinFlipEngine, delta = 0.01, debug = False, benchmark=False, benchmark_flips=int(1e4)) -> list[CoinFlipEngine]:
    if debug and input_model.name is not None:
        print(f'Coin is {input_model.name}')

    calix_output = Calix(output_data_sequence, input_model.markov, 
                         input_model.memory_depth, input_model.size, 
                         delta, debug, benchmark_flips)
    # freakbob_output = Freakbob(input_model, output_data_sequence, input_model.size, 
    #                            input_model.memory_depth, delta, debug, benchmark_flips)
    sleeper_output = Sleeper(output_data_sequence, input_model.memory_depth, 
                             input_model.size, debug)
    
    models = [ calix_output, sleeper_output ]

    return models

def Calix(output_data_sequence: list[int], markov: list[dict], memory_depth: int, size: int, delta, debug, benchmark_flips) -> CoinFlipEngine:
    """
    Calix is the heavily informed system:

    (1) informed about memory depth and markov rules
    (2) works only on square systems
    """
    coin_output_histogram = np.zeros(shape=(size, size)) 
    current_coin = 0
    memory = []
    num_flips = len(output_data_sequence)

    if debug:
        print('Beginning Calix Mk. 1 Estimation')

    for epoch, output in enumerate(output_data_sequence):
        memory.append(output)
        coin_output_histogram[current_coin][output] += 1

        if len(memory) == memory_depth:
            next_coin = markov[current_coin][tuple(memory)]
            current_coin = next_coin
            memory = [] 

        if debug and (epoch + 1) % 2500 == 0: 
            print(f"Epoch {epoch + 1}/{num_flips}.")

    row_sums = coin_output_histogram.sum(axis=1, keepdims=True)
    # to avoid divison by zero
    row_sums[row_sums == 0] = 1

    if debug and 0 in coin_output_histogram:
        print('Error: Coin output histogram contains zero for an output.')
        print(coin_output_histogram)
        print(markov)

    return CoinFlipEngine(
        probabilities = (coin_output_histogram / row_sums) * 100,
        markov = markov,
        memory_depth = memory_depth,
        size = size,
        benchmark = True,
        benchmark_flips = benchmark_flips)

def Freakbob(input_model, output_data_sequence, size, memory_depth, delta, debug, benchmark_flips):
    """
    Freakbob is a lightly informed system, knowing only the size and memory depth 
    but not the markov rules.

    It generates all possible markov arrays and reverse engineers on each, then
    selects the model with the lowest error.

    However, this solution is incredibly slow! Generating all possible markov
    arrays is already an incredibly slow process, combined with reverse engineering
    on every possible one.
    """

    if debug:
        print('Beginning Freakbob Estimation')

    all_possible_markovs = generate_all_possible_markov_system(size, memory_depth)
    num_markovs = len(all_possible_markovs)

    best_guess = None
    best_guess_error = np.inf

    for i, markov in enumerate(all_possible_markovs):
        if debug:
            print(f'Freakbob Estimation {i+1}/{num_markovs}')
        guess = Calix(output_data_sequence, markov, memory_depth, size, delta, debug, benchmark_flips)
        error = calculate_model_error(input_model, guess)

        if error < best_guess_error:
            best_guess = guess
            best_guess_error = error

    return best_guess
