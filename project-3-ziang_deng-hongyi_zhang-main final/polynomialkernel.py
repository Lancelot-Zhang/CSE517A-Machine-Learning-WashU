import numpy as np

def polynomial_kernel(x1, x2, kpar):
    """
    Polynomial kernel.
    
    Parameters:
        x1: array-like, has dimensions (d,n)
        x2: array-like, has dimensions (d,n)
        kpar: int, degree of the polynomial

    Returns:
        K: the kernel matrix; array-like, has dimensions (n,n)
    """
    # YOUR CODE HERE

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    poly_kernel = (x1.T @ x2 + 1) ** kpar

    return poly_kernel
