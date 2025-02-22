import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import sys
sys.path.insert(1, '../')
from coins import CoinFlipEngine
from generator import generate_possible_combinations

class MultiOutputRNN(nn.Module):
    def __init__(self, input_size, system_size, hidden_size, sequence_length, num_sequences):
        super().__init__()
        self.input_size      = input_size
        self.system_size     = system_size
        self.hidden_size     = hidden_size
        self.num_sequences   = num_sequences
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # predicts the current coin
        self.ff_coin = nn.Linear(hidden_size, system_size)

        # predicts the next state
        self.ff_state = nn.Linear(hidden_size, system_size)

        # predicts the probability distribution
        self.ff_prob  = nn.Linear(hidden_size, num_sequences)

    def forward(self, training_data_batch):
        print('shape:', training_data_batch.shape, 'seq length:', self.sequence_length)
        # training_data_batch = training_data_batch.unsqueeze(-1)
        initial_hidden_state = torch.zeros(self.input_size, 
                                           training_data_batch.size(0), 
                                           self.hidden_size)
        print()
        print('shape:', training_data_batch.shape, 'seq length:', self.sequence_length)
        print()
        current_hidden_state, _ = self.rnn(training_data_batch, initial_hidden_state)
        current_hidden_state = current_hidden_state[:, -1, :]
        
        # generate predictions
        coin_pred = self.ff_coin(current_hidden_state)
        state_pred = self.ff_state(current_hidden_state)
        probability_pred = torch.softmax(self.ff_prob(current_hidden_state), dim=1)

        # return predictions
        return coin_pred, state_pred, probability_pred

def extract_estimations(model: MultiOutputRNN, size: int, memory_depth: int, possible_sequences: list[tuple[int]]) -> tuple[np.array, list[dict]]:
    """
    Extracts the estimations in the probability array and 
    markov from the model's current state
    """
    model.eval()

    # generate default markov
    inferred_markov = [ { seq: 0 for seq in possible_sequences } for _ in range(size) ] 

    # generate default
    inferred_probabilities = np.zeros(shape=(size, size))

    with torch.no_grad():
        for seq in possible_sequences:
            coin_pred, state_pred, prob_pred = model(seq)
            
            inferred_coin = coin_pred.argmax(dim=1).item()
            inferred_next_coin = state_pred.argmax(dim=1).item()
            inferred_prob = prob_pred.squeeze().tolist()

            inferred_markov[inferred_coin][seq] = inferred_next_coin
            inferred_probabilities[inferred_next_coin] = inferred_prob

    return inferred_probabilities, inferred_markov

def Sleeper(output_data_sequence: list[int], coin_state_sequence: list[int], memory_depth: int, size: int, debug: bool, epochs=100) -> CoinFlipEngine:
    """
    Sleeper is lightly informed, determining the markov and the original probabilities
    simultaneously via the use of supervised machine learning in the form of a Recurrent
    Neural Network.

    For training purposes, however, it is also given the series of coin states
    """
    if debug:
        print('Beginning Sleeper Estimation')

    num_flips = len(output_data_sequence)
    possible_sequences = generate_possible_combinations(size, memory_depth)

    # generate dataset for training
    input_data = []
    targets = dict(coin=[], state=[], probability=[])

    for i in range(num_flips-1):
        current_output = output_data_sequence[i]
        current_coin = coin_state_sequence[i]
        next_coin = coin_state_sequence[i+1]
        probability = 

        input_data.append(output_data_sequence[i:i + memory_depth])
        targets.append(output_data_sequence[i + memory_depth - 1])

    input_data = torch.tensor(input_data, dtype=torch.float32)
    for v in targets:
        targets[v] = torch.tensor(targets[v], dtype=torch.float32)

    # init model
    model = MultiOutputRNN(input_size = 1,
                           system_size = size,
                           hidden_size = 8,
                           sequence_length = memory_depth,
                           num_sequences = len(possible_sequences))

    """
    We're optimizing the model iteratively, so we define error purely in the 
    cross entropy loss at each step rather than the classical definition we've
    given for entire models.
    """
    criterion_coin = nn.CrossEntropyLoss()
    criterion_seq = nn.CrossEntropyLoss()
    criterion_prob = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        coin_preds, state_preds, prob_preds = model(input_data)

        loss_coin = criterion_coin(coin_preds, )

        loss.backward()
        optimizer.step()

        if debug and epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    est_probabilities, est_markov = extract_estimations(model, size, 
                                                        memory_depth, 
                                                        possible_sequences)
    
    return CoinFlipEngine(
        probabilities = est_probabilities,
        markov = est_markov,
        memory_depth = memory_depth,
        size = size,
        benchmark = True
    )

