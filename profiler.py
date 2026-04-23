def compute_score(error, epsilon, alpha=1.0, beta=1.0):

    utility = 1 / (1 + error)   # higher when error is low
    privacy_loss = epsilon      # higher epsilon = worse privacy

    score = alpha * utility - beta * privacy_loss

    return score