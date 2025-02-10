import numpy as np
from matplotlib import pyplot

THEOR_COLORS = ['red', 'green', 'blue', 'orange', 'black', 'purple', 'magenta']

def plot_coin_convergence(num_flips, num_outputs, percent_history, theoretical_distribution, coin_label):
  pyplot.figure()
  pyplot.axis([0, num_flips, 0, 100])
  pyplot.title(f'{coin_label} - Probability Convergence')
  pyplot.ylabel('Output Percentage (%)')
  pyplot.xlabel('Number of Flips')

  for output_index in range(0, num_outputs):
    # Plot the theoretical distribution to converge to
    pyplot.axhline(theoretical_distribution[output_index],
                   label=f"Output {output_index} (Theoretical) - {theoretical_distribution[output_index]}%",
                   linestyle='solid', color=THEOR_COLORS[output_index])

    # Plot the percent history data
    pyplot.plot(np.arange(0, num_flips + 1, 1),
                percent_history[output_index],
                label=f"Output {output_index} (Empirical)",
                linestyle='solid',
                linewidth=2,
                markersize=6,
                alpha=0.5)

  pyplot.legend()
  pyplot.show()
