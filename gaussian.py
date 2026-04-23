import numpy as np

def exponential_mechanism(data, candidates, epsilon, sensitivity=1.0):

    # Score function: example → count of elements <= candidate
    def score_function(h):
        return np.sum(data <= h)

    # Compute scores
    scores = np.array([score_function(h) for h in candidates])

    # Normalize (optional but good practice)
    scores = scores / len(data)

    # Compute probabilities
    scaled_scores = (epsilon * scores) / (2 * sensitivity)
    scaled_scores = scaled_scores - np.max(scaled_scores)  # stability

    exp_scores = np.exp(scaled_scores)
    probabilities = exp_scores / np.sum(exp_scores)

    # Sample one output
    selected = np.random.choice(candidates, p=probabilities)

    return selected, probabilities