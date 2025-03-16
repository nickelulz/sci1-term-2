import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm

# Define weather state labels
weather_labels = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Rain", "Heavy Rain"]

# Function to load weather categories from a file
def load_weather_data(filepath):
    """Reads newline-separated weather categories and returns a NumPy array."""
    with open(filepath, "r") as f:
        return np.array([int(line.strip()) for line in f if line.strip()]).reshape(-1, 1)

# Load UBC and Richmond data
ubc_data = load_weather_data("data/ubc-train.csv")
richmond_data = load_weather_data("data/richmond-test.csv")

# Function to split data into 75% train, 25% test
def split_data(data):
    split_idx = int(len(data) * 0.75)
    return data[:split_idx], data[split_idx:]

# Split UBC and Richmond data
ubc_train, ubc_test = split_data(ubc_data)
richmond_train, richmond_test = split_data(richmond_data)

# Define the HMM model with smoothing
n_states = 6
min_prob = 1e-2  # Small probability floor for transitions

def train_hmm(train_data):
    """Creates a new HMM instance and fits it to the training data with smoothing."""
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=200, tol=1e-4, verbose=True, init_params="ste")
    
    # Initialize transition and emission probabilities with pseudocounts
    model.startprob_ = np.full(n_states, 1.0 / n_states)
    model.transmat_ = np.full((n_states, n_states), min_prob)  # Start with uniform probabilities
    np.fill_diagonal(model.transmat_, 1 - (n_states - 1) * min_prob)  # Ensure valid transitions

    model.fit(train_data)
    return model

# Train separate HMM models for each case
ubc_model = train_hmm(ubc_train)  # UBC → UBC
richmond_model = train_hmm(richmond_train)  # Richmond → Richmond
cross_model = train_hmm(ubc_train)  # UBC → Richmond (cross-test)

# Function to evaluate an HMM model on test data
def evaluate_model(model, test_data, test_name):
    log_likelihood = model.score(test_data)
    per_symbol_log_likelihood = log_likelihood / len(test_data)
    print(f"\n[{test_name}]")
    print(f"  Total Log Likelihood: {log_likelihood:.4f}")
    print(f"  Per-Symbol Log Likelihood: {per_symbol_log_likelihood:.6f}")

    # Predict using the Viterbi algorithm
    def predict_next_state(model, sequence):
        log_prob, hidden_states = model.decode(sequence, algorithm="viterbi")
        return hidden_states[-1]  # Last predicted state

    predicted_states = [predict_next_state(model, test_data[:i+1]) for i in range(len(test_data)-1)]
    actual_states = test_data[1:].flatten()

    # Compute accuracy
    accuracy = np.mean(np.array(predicted_states) == actual_states)
    print(f"  Prediction Accuracy: {accuracy:.2%}")
    return accuracy

# Evaluate models
ubc_acc = evaluate_model(ubc_model, ubc_test, "UBC (Trained on UBC)")
richmond_acc = evaluate_model(richmond_model, richmond_test, "Richmond (Trained on Richmond)")
cross_acc = evaluate_model(cross_model, richmond_test, "Richmond (Trained on UBC)")

# Function to plot transition matrices separately
def plot_transition_matrix(model, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(model.transmat_, annot=True, cmap="Blues", fmt=".2f",
                xticklabels=weather_labels, yticklabels=weather_labels)
    plt.title(title)
    plt.xlabel("Next State")
    plt.ylabel("Current State")

# Plot transition matrices in separate windows
plot_transition_matrix(ubc_model, "UBC Transition Matrix")
plot_transition_matrix(richmond_model, "Richmond Transition Matrix")
plot_transition_matrix(cross_model, "UBC → Richmond Transition Matrix")

# Plot weather distributions separately
plt.figure(figsize=(6, 5))
plt.hist(ubc_data.flatten(), bins=np.arange(7)-0.5, alpha=0.5, label="UBC", density=True, color="blue")
plt.hist(richmond_data.flatten(), bins=np.arange(7)-0.5, alpha=0.5, label="Richmond", density=True, color="red")
plt.xticks(range(6), weather_labels, rotation=45)
plt.title("Weather Category Distributions in UBC vs Richmond")
plt.legend()
plt.show()

