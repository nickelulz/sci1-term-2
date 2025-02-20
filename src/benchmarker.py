import numpy as np

def calculate_probability_error(probability_a, probability_b):
    """
    Calculates the error between the two prrobability probabilitys
    """

    return np.sum(np.abs(probability_a - probability_b))

def calculate_markov_error(memory_depth_a, memory_depth_b, markov_a, markov_b, size):
    """
    Calculates the error between two markov matrices by counting the number of differences
    and dividing by the size -- also probably to be informed by the memory depth as well
    """
    return 1 # TODO
    

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

def calculate_distribution_error(dist_a, dist_b):
    return np.sum(np.abs(dist_a-dist_b))

def calculate_model_error(input_model, output_model, input_flip_history, output_flip_history) -> float:
    """
    Compares two CoinFlipEngine models by calculating the error 
    of one model against the other.
    """
    if input_model.size != output_model.size:
        return np.inf
    
    distribution_error = calculate_distribution_error(input_flip_history, output_flip_history)
    probability_error = calculate_probability_error(input_model.probabilities, 
                                                      output_model.probabilities)
    complexity_error = calculate_complexity_error(input_model.complexity, output_model.complexity)
    markov_error = calculate_markov_error(input_model.memory_depth, output_model.memory_depth, 
                                          input_model.markov, output_model.markov, input_model.size)
        
    total_error = probability_error * markov_error * distribution_error
    #if (total_error > 12):
    #    print(f"\nHigh Error Detected, input model: {input_model}, output model: {output_model}")
    return total_error
