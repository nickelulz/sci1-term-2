import numpy as np

def calculate_accuracy(distribution_mse, complexity_pe):
  """
  Calculates the accuracy index between two models
  """
  return distribution_mse * (complexity_pe * 100)

"""
The error tolerance for how far each of the empirical
distributions can be from the expected distribution.
"""
BENCHMARK_ERROR_TOLERANCE = 1e-2

def benchmark_distribution_error(distribution_a, distribution_b):
  """
  Calculates the Mean-Squared Error between the two distributions
  CONSTRAINT: must be the same size
  """
  return (np.sum(np.abs(distribution_a - distribution_b) ** 2) /
          len(distribution_a))

def calculate_model_error(input_model, output_model) -> float:
    """
    Compares two CoinFlipEngine models by calculating the accuracy
    of one model against the other.
    """
    if input_model.size != output_model.size:
        return np.inf

    distribution_error = benchmark_distribution_error(input_model.theoretical_distribution, 
                                                      output_model.theoretical_distribution)
    
    complexity_percent_error = 0

    if input_model.complexity == 0 or output_model.complexity == 0:
        complexity_percent_error = max(output_model.complexity, input_model.complexity)
    else:
        complexity_percent_error = (np.abs(input_model.complexity - output_model.complexity) / input_model.complexity)

    return calculate_accuracy(distribution_error, complexity_percent_error)
