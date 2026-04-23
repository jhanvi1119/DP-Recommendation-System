import numpy as np

def gaussian_mechanism(true_value, sensitivity, epsilon, delta=1e-5):

    # Compute sigma (standard deviation)
    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon

    # Add Gaussian noise
    noise = np.random.normal(0, sigma)
    noisy_value = true_value + noise

    return noisy_value