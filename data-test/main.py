import numpy as np
from matplotlib import pyplot as plt
from ajani import Ajani

def load_weather_data(filepath):
    """
    Reads newline-separated weather categories and returns a NumPy array.
    """
    with open(filepath, "r") as f:
        return np.array([int(line.strip()) for line in f if line.strip()]).reshape(-1, 1).flatten().tolist()

def load_mpox_data(filepath):
    # read data from dat to a map of lists
    form = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split(':')
            countryname = tokens[0]
            values = [int(x) for x in tokens[1].split(',')]
            # we have to replace all -1s with 2
            values = [2 if x == -1 else x for x in values]
            form[countryname] = values
    return form

def split_data(data, mpox=False):
    split_idx = int(len(data) * 0.75)
    return data[:split_idx], data[split_idx:]

TOTAL_SAMPLES = 100

def main():
    # Load UBC and Richmond data
    ubc_data = load_weather_data("data/ubc-train.csv")
    richmond_data = load_weather_data("data/richmond-test.csv")
    mpox_data = load_mpox_data('data/mpox-data-sanitized.dat')

    # Split datasets
    ubc_train, ubc_test = split_data(ubc_data)
    richmond_train, richmond_test = split_data(richmond_data)

    # Train & Evaluate
    print("[UBC Model]")
    ubc_accuracies = []
    for i in range(TOTAL_SAMPLES):
        ubc_model, ubc_accuracy = Ajani(ubc_train, ubc_test, size=6, memory_depth=2, num_samples=50)
        print(f'UBC Model {i+1}/{TOTAL_SAMPLES} Accuracy: {ubc_accuracy*100:.1f}')
        ubc_accuracies.append(ubc_accuracy)

    """
    print('[Richmond Model]')
    richmond_model, richmond_accuracy = Ajani(richmond_train, richmond_test, size=6, memory_depth=2, num_samples=50)
    print(f'Richmond Model Accuracy: {richmond_accuracy*100:.1f}')
    """

    mpox_world_data = mpox_data['World']
    mpox_world_train, mpox_world_test = split_data(mpox_world_data)
    print('[Mpox World Model]')
    mpox_accuracies = []

    for i in range(TOTAL_SAMPLES):
        mpox_world_model, mpox_world_accuracy = Ajani(mpox_world_train, mpox_world_test, size=3, memory_depth=1, num_samples=500)
        print(f'mpox {i+1}/{TOTAL_SAMPLES} Model Accuracy: {mpox_world_accuracy*100:.1f}')
        mpox_accuracies.append(mpox_world_accuracy)

    avg_ubc_accuracy = np.mean(np.array(ubc_accuracies))
    avg_mpox_accuracy = np.mean(np.array(mpox_accuracies))

    print(f'Average UBC Accuracy: {avg_ubc_accuracy*100:.2f}')
    print(f'Average mpox accuracy: {avg_mpox_accuracy*100:.2f}')
    print(f'Difference (mpox-UBC): {(avg_mpox_accuracy-avg_ubc_accuracy)*100:.2f}')

if __name__ == '__main__':
    main()

