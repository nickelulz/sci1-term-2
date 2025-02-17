import numpy as np

def calculate_distribution_error(distribution_a, distribution_b):
  """
  Calculates the error between the two prrobability distributions
  """
  return np.sum(np.abs(distribution_a - distribution_b))

def calculate_complexity_error(complexity_a, complexity_b):
    """
    Calculates the percent error (or absolute error, if one or both is zero) 
    between two models
    """
    complexity_pe = -1

    if complexity_a == 0 or complexity_b == 0:
        complexity_pe = max(complexity_a, complexity_b)
    else:
        complexity_pe = (np.abs(complexity_a - complexity_b) / complexity_a)

    return complexity_pe

def calculate_model_error(input_model, output_model) -> float:
    """
    Compares two CoinFlipEngine models by calculating the error 
    of one model against the other.
    """
    if input_model.size != output_model.size:
        return np.inf

    distribution_mse = calculate_distribution_error(input_model.probabilities / 100, 
                                                    output_model.probabilities / 100)
    complexity_pe = calculate_complexity_error(input_model.complexity,
                                               output_model.complexity)

    return distribution_mse * complexity_pe
