import joblib
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from coins import CoinFlipEngine
from generator import generate_random_markov, generate_random_model
from error import calculate_model_error

from hmmlearn import hmm
import numpy as np

def train_sleeper():
    # generate 1000 coins randomly
    num_coins = 10
    num_flips = 1e4
    size_range = (1,2)
    memory_depth_range = (1,1)

    epochs = 10

    # model
    model = hmm.MultinomialHMM(n_components=2, 
                               n_iter=epochs, 
                               random_state=42)

    # train on every coin
    for i in range(num_coins):
        input_model = generate_random_model(size_range,
                                            memory_depth_range,
                                            benchmark=True,
                                            benchmark_flips=int(num_flips))
        flip_data = input_model.benchmark_result.flip_history
        hidden_states = input_model.benchmark_result.state_data

        num_flips = len(flip_data)
        padding = int(num_flips % input_model.memory_depth)
        newlength = num_flips - padding
        sequences = np.array(flip_data[padding:]).reshape(int(newlength / input_model.memory_depth), input_model.memory_depth)
        hidden_states = hidden_states[padding:]

        one_hot_sequences = np.zeros((sequences.shape[0], sequences.shape[1], 2))

        for i, seq in enumerate(sequences):
            for j, flip in enumerate(seq):
                one_hot_sequences[i, j, flip] = 1

        one_hot_sequences = one_hot_sequences.reshape(-1, 2)

        model.fit(one_hot_sequences)

        predicted_hidden_states = model.predict(sequences)
        prediction_error = np.mean(predicted_hidden_states != hidden_states)

    joblib.dump(model, 'hmm_model.pkl')

if __name__ == '__main__':
    train_sleeper()
