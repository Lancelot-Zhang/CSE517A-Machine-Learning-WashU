from numpy import maximum
import numpy as np
'''
    INPUT:
    xTr:    dxn matrix - 2d numpy array (each column is an input vector)
    yTr:    1xn vector - 2d numpy array (each entry is a label)
   lambdaa: float (regularization constant)
    w:      dx1 weight vector - 2d numpy array (default w=0)

    OUTPUTS:

    reg_loss:      float (the total regularized loss obtained with w on xTr and yTr)
    gradient:  dx1 vector - 2d numpy array (the gradient at w)

    [d,n]=size(xTr);
'''
def hinge(w,xTr,yTr,lambdaa):

    # YOUR CODE HERE

    """
    Calculate the hinge loss and its gradient
    """

    # Calculate the hinge loss
    reg_loss = np.sum(np.maximum(0, 1 - yTr * np.dot(w.T, xTr))) + lambdaa * np.dot(w.T, w)

    # Calculate the gradient
    gradient = -np.dot(xTr, np.where(yTr * np.dot(w.T, xTr)> 1, 0, yTr).T) + 2 * lambdaa * w

    return reg_loss,gradient
