import numpy as np

def weights_to_ranges(probability_array):
  """
  Converts weight arrays to range benchmarker arrays
  """
  probability_array = probability_array / 100
  ranges = np.zeros(probability_array.shape)
  ranges[0] = probability_array[0]
  for i in range(1, len(probability_array)):
    ranges[i] = ranges[i-1] + probability_array[i]
  return ranges

def interpret_individual_markov(markov_arr_coin: np.array) -> dict:
    markov = {}
    for output, next_coin in enumerate(markov_arr_coin):
        markov[tuple([output])] = next_coin
    return markov

def interpret_simple_markov(markov_arr) -> list[dict]:
    return [interpret_individual_markov(markov) for markov in markov_arr]
