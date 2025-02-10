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

def generate_random_distribution(n):
    """
    Generates a random distribution of n outputs that sums to 100.
    """
    random_values = np.random.rand(n) 
    normalized_values = random_values / random_values.sum()
    distribution = (normalized_values * 100).round().astype(int)

    diff = 100 - distribution.sum()
    for _ in range(abs(diff)):
        index = np.random.choice(n)
        distribution[index] += 1 if diff > 0 else -1

    return np.array(distribution.tolist())
