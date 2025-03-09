import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from coins import CoinFlipEngine
from generator import generate_random_markov 

from hmmlearn import hmm
import numpy as np

from dataclasses import dataclass

@dataclass
class SleeperEstimation():
    probabilities: np.array
    transition_matrix: np.array
    size: int
    empirical_distribution: np.array

def Sleeper(output_data_sequence: list[int], 
            memory_depth: int, size: int, debug: bool = False, epochs=100):
    data = np.tile(np.array(output_data_sequence), 5).reshape(-1, 1)
    print(np.shape(data))

    model = hmm.MultinomialHMM(n_components=size,
                               n_iter=epochs, 
                               verbose=debug,
                               init_params="")
    model.n_features = size

    starting_probabilities = np.random.rand(size, size)
    starting_probabilities = starting_probabilities / np.sum(starting_probabilities, axis=1, keepdims=True)

    starting_transmat = np.random.rand(size, size)
    starting_transmat = starting_transmat / np.sum(starting_transmat, axis=1, keepdims=True)

    starting_emissions = np.random.rand(size, size)
    starting_emissions = starting_emissions / np.sum(starting_emissions, axis=1, keepdims=True)

    print(starting_probabilities, '\n', starting_transmat, '\n', starting_emissions)

    model.startprob_ = starting_probabilities
    model.transmat_ = starting_transmat 
    model.emissionprob_ = starting_emissions

    print(model.emissionprob_)
    model.fit(data)
    print(model.emissionprob_)
    log_prob, hidden_states = model.decode(data, algorithm='viterbi')

    # calculate the standing dist.
    eigvals, eigvec = np.linalg.eig(model.transmat_.T)
    stationary_dist = eigvec[:, np.isclose(eigvals, 1)].real
    # normalize
    stationary_dist = stationary_dist / stationary_dist.sum()

    if debug:
        print('======================================================================')
        print(f'Size: {size}, Memory Depth: {memory_depth}')
        print(f'Hidden States: {hidden_states}')
        print(f'Coins Reached: {set(hidden_states)}') 
        print(f'Estimated Probabilties: {model.emissionprob_}')
        print(f'Transition Matrix')
        print(model.transmat_)
        print(f'Standing Distribution: ', stationary_dist)
        print('======================================================================')

    return SleeperEstimation(probabilities=model.emissionprob_,
                             transition_matrix=model.transmat_,
                             size = size,
                             empirical_distribution = stationary_dist) 
