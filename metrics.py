import numpy as np
from mechanism.laplace import laplace_mechanism
from mechanism.randomized_response import randomized_response
from mechanism.exponential import exponential_mechanism
from mechanism.gaussian import gaussian_mechanism
from utils.metrics import compute_score


def evaluate_numerical_mechanisms(data, epsilon, delta=1e-5):
   
    results = {}

    true_value = np.mean(data)

    sensitivity = (np.max(data) - np.min(data)) / len(data)

    laplace_result = laplace_mechanism(true_value, sensitivity, epsilon)
    laplace_error = abs(true_value - laplace_result)
    laplace_score = compute_score(laplace_error, epsilon)

    results["laplace"] = {
        "noisy_value": laplace_result,
        "error": laplace_error,
        "score": laplace_score
    }

    gaussian_result = gaussian_mechanism(true_value, sensitivity, epsilon, delta)
    gaussian_error = abs(true_value - gaussian_result)
    gaussian_score = compute_score(gaussian_error, epsilon)

    results["gaussian"] = {
        "noisy_value": gaussian_result,
        "error": gaussian_error,
        "score": gaussian_score
    }

    return true_value, results

def evaluate_binary_mechanism(data, epsilon):

    results = {}

    true_mean = np.mean(data)

    noisy_data, estimated_mean = randomized_response(data, epsilon)
    error = abs(true_mean - estimated_mean)

    score = compute_score(error, epsilon)

    results["randomized_response"] = {
        "noisy_value": estimated_mean,
        "error": error,
        "score": score
    }

    return true_mean, results

def evaluate_selection_mechanism(data, epsilon):

    results = {}

    # Candidates (can adjust later)
    candidates = np.linspace(np.min(data), np.max(data), 20)

    selected, probs = exponential_mechanism(data, candidates, epsilon)

    # For evaluation → compare with max value
    true_value = np.max(data)
    error = abs(true_value - selected)

    score = compute_score(error, epsilon)

    results["exponential"] = {
        "selected_value": selected,
        "error": error,
        "score": score
    }

    return true_value, results