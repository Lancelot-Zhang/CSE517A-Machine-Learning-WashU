
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

def ridge(w,xTr,yTr,lambdaa):

    # YOUR CODE HERE

    # Calculate the regularized loss
    reg_loss = np.sum((np.dot(w.T,xTr)-yTr)**2) + lambdaa * np.linalg.norm(w)**2

    # Calculate the gradient
    gradient = 2 * np.dot(xTr, (np.dot(xTr.T,w)-yTr.T)) + 2 * lambdaa * w

    return reg_loss,gradient
