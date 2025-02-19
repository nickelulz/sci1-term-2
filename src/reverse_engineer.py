import numpy as np
from coins import CoinFlipEngine

def reverse_engineer_model(output_data_sequence: list[int], input_model: CoinFlipEngine, delta = 0.01, output = False) -> list[CoinFlipEngine]:
    if input_model.name is not None:
        print(f'Coin is {input_model.name}')
    calix_output = Calix(output_data_sequence, input_model.markov, input_model.memory_depth, input_model.size, delta, output)
    
    return [ calix_output ]

def Calix(output_data_sequence: list[int], markov: list[dict], memory_depth: int, size: int, delta, output) -> CoinFlipEngine:
    """
    Calix is the heavily informed system:

    (1) informed about memory depth and markov rules
    (2) works only on square systems
    """
    observations = np.zeros(shape=(size, size))
    
    # start storing values in the first coin by default
    current_coin = 0
    epoch = 0

    if output:
        print('Beginning Calix Mk. 1 Estimation')

    memory = []

    for flip in output_data_sequence:
        memory.append(flip)
        observations[current_coin][flip] += 1

        if len(memory) == memory_depth:
            next_coin = markov[current_coin][tuple(memory)]

            current_coin = next_coin
            memory = [] 

        epoch += 1

        if epoch % 2500 == 0 and output: 
            print(f"Current Observations: {observations}")

    row_sums = observations.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1

    if 0 in observations and output:
        print('zero in obs')
        print(observations)
        print(markov)

    return CoinFlipEngine(
        probabilities = (observations / row_sums) * 100,
        markov = np.array(markov),
        size = size,
        initial_coin_index = 0)

def Freakbob(output_data_sequence, size, memory_depth, delta, output):
    """
    Freakbob is a lightly informed system, knowing only the size and memory depth but not the markov rules
    """
    raise NotImplementedError()
