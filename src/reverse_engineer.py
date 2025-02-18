import numpy as np
from coins import CoinFlipEngine
from hash_table import HashTable
from util import markov_next

def reverse_engineer_model(output_data_sequence: list[int], input_model: CoinFlipEngine, delta = 0.01, output = False) -> list[CoinFlipEngine]:
    calix_output = Calix(output_data_sequence, input_model.markov, input_model.memory_depth, input_model.size, delta, output)
    
    return [ calix_output ]

def Calix(output_data_sequence: list[int], markov: list[HashTable], memory_depth: int, size: int, delta, output) -> CoinFlipEngine:
    """
    Calix is the heavily informed system:

    (1) informed about memory depth and markov rules
    (2) works only on square systems
    """
    estimated_model = np.random.rand(size, size) * 100
    observations = np.zeros(shape=(size, size))
    
    # start storing values in the first coin by default
    current_coin = 0
    epoch = 0

    if output:
        print('Beginning Calix Mk. 1 Estimation')

    for flip in output_data_sequence:
        next_coin = markov_next(markov, current_coin, memory)
        observations[current_coin][next_coin] += 1
        current_coin = next_coin

        row_sums = observations.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        gradients = (observations / row_sums) * 100 - estimated_model
        estimated_model += delta * gradients

        epoch += 1

        if epoch % 2500 == 0 and output: 
            print(f"Current Estimated Model: {estimated_model}")
            print(f"Current Observations: {observations}")
            print(f"Current Observations Markov: {(observations / row_sums)*100}")

    return CoinFlipEngine(
        probabilities = estimated_model,
        markov = np.array(markov),
        size = size,
        initial_coin_index = 0)

def Freakbob(output_data_sequence, size, memory_depth, delta, output):
    """
    Freakbob is a lightly informed system, knowing only the size and memory depth but not the markov rules
    """
    raise NotImplementedError()
