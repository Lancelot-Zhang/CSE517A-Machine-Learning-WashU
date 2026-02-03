import numpy as np

def l2_distance(X, Z):
    """
    Compute the L2 (Euclidean) distance between two vectors.

    Parameters:
        X: dxn data matrix with n vectors (columns) of dimensionality d
        Z: dxm data matrix with m vectors (columns) of dimensionality d

    Returns:
        Matrix D of size nxm 
        D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
    """
    # YOUR CODE HERE

    X_sq = np.sum(X**2, axis=0) # sum of squares of each column
    Z_sq = np.sum(Z**2, axis=0) # sum of squares of each column
    
    cross = np.dot(X.T, Z) # dot product of X and Z

    D_sq = X_sq[:, np.newaxis] + Z_sq[np.newaxis, :] - 2 * cross # calculate the squared distance
    D = np.sqrt(np.maximum(D_sq, 0)) # take the square root of the squared distance

    # D_sq = np.add.outer(X_sq, Z_sq) - 2 * cross
    # D_sq = np.clip(D_sq, 0, None)
    # D = np.sqrt(D_sq)

    return D