import numpy as np
from coins import CoinFlipEngine

def Calix(output_data_sequence, markov, size, delta=0.01, output=False):
    # n = number of outputs
    estimated_model = np.random.rand(size, size) * 100
    observations = np.zeros(shape=(size, size))
    
    # start storing values in the first coin by default
    current_coin = 0
    epoch = 0

    for flip in output_data_sequence:
        next_coin = markov[current_coin][flip]
        observations[current_coin][next_coin] += 1
        curren_coin = next_coin

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
