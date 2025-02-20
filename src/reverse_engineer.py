import numpy as np
from coins import CoinFlipEngine

def reverse_engineer_model(output_data_sequence: list[int], input_model: CoinFlipEngine, delta = 0.01, debug = False, benchmark=False, benchmark_flips=int(1e4)) -> list[CoinFlipEngine]:
    if debug and input_model.name is not None:
        print(f'Coin is {input_model.name}')

    calix_output = Calix(output_data_sequence, input_model.markov, input_model.memory_depth, input_model.size, delta, debug)
    
    models = [ calix_output ]

    if benchmark:
        for model in models:
            model.benchmark(benchmark_flips)

    return models

def Calix(output_data_sequence: list[int], markov: list[dict], memory_depth: int, size: int, delta, debug) -> CoinFlipEngine:
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

    if debug and 0 in observations:
        print('Error: Coin output histogram contains zero for an output.')
        print(coin_output_histogram)
        print(markov)

    return CoinFlipEngine(
        probabilities = (coin_output_histogram / row_sums) * 100,
        markov = np.array(markov),
        memory_depth = memory_depth,
        size = size)

def Freakbob(output_data_sequence, size, memory_depth, delta, output):
    """
    Freakbob is a lightly informed system, knowing only the size and memory depth but not the markov rules
    """
    raise NotImplementedError()
