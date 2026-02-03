import numpy as np

def negative_log_predictive_density(y_tests, y_preds, variances):
    """
    Compute the negative log predictive density (NLPD) using the standardized data.
    """
    # YOUR CODE HERE

    square_terms = 0.5 * ((y_tests - y_preds) ** 2) / variances
    log_terms = 0.5 * np.log(2 * np.pi * variances)
    nlpd = np.mean(log_terms + square_terms)

    return nlpd