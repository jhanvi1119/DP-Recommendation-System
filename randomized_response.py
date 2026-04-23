import numpy as np

def laplace_mechanism(true_value, sensitivity, epsilon):

    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    noisy_value = true_value + noise

    return noisy_value