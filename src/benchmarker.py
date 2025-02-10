"""
A max limit to avoid an infinite loop.
"""
ITERATION_MAX_LIMIT = 1e10

def benchmark_model_convergence(model):
  """
  Benchmarks the converge speed of the model in the
  number of coin flips required for the model to
  converge within 10^-2 of the theoretical distribution.

  Called natively within the constructor of the CoinFlipEngine,
  and stored in CoinFlipEngine::model_convergence_benchmark.
  """
  flip_histogram = [0] * coin.number_of_outputs
  number_of_coin_flips = 0

  def histogram_to_probability_disribution(histogram, number_of_coin_flips):
    return (np.array(histogram) / number_of_coin_flips) * 100

  while (benchmark_distribution_error(
          self.histogram_to_probability_distribution(flip_histogram, number_of_coin_flips),
          model.theoretical_distribution) > BENCHMARK_ERROR_TOLERANCE
         and number_of_coin_flips < ITERATION_MAX_LIMIT):

    number_of_coin_flips += 1
    flip_result = coin.flip()
    flip_histogram[flip_result] += 1

  empirical_distribution = histogram_to_probability_disribution(flip_histogram)

  return number_of_coin_flips, empirical_distribution


def calculate_accuracy(distribution_mse, complexity_pe, ):
  """
  Calculates the accuracy index between two models
  """
  raise NotImplementedError()

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

def compare_models(input_model, output_model) -> float:
    """
    Compares two CoinFlipEngine models by calculating the accuracy
    of one model against the other.
    """
    if expected_model.size != generated_model.size:
        return 0

    distribution_error = benchmark_distribution_error(input_model.theoretical_distribution, 
                                                      output_model.theoretical_distribution)
    complexity_error = (np.abs(input_model.complexity - output_model.complexity) / output_model.model_convergence_benchmark)

    return calculate_accuracy(distribution_error, convergence_percent_error)
