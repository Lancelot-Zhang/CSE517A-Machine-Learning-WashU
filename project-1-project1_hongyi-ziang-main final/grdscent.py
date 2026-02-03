import numpy as np

def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 # minimum step size for gradient descent

    # YOUR CODE HERE

    w = w0.copy()

    gamma_inc = 1.01
    gamma_dec = 0.5
    alpha = stepsize
    
    for i in range(maxiter):
        f, g = func(w)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tolerance:
            break

        w_new = w - alpha * g
        f_new, _ = func(w_new)

        if f_new < f:
            alpha = min(alpha * gamma_inc, 1)
        else:
            alpha = max(alpha * gamma_dec, eps)
        
        w = w_new

    return w
