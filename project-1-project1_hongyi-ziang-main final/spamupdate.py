import numpy as np

def spamupdate(w,email,truth):

    # Input:
    # w     weight vector
    # email instance vector
    # truth label
    #
    # Output:
    #
    # updated weight vector
    #
    # INSERT CODE HERE:

    # # Calculate the score
    # score = np.dot(w.T, email)
    
    # # Transform the score into predicted label
    # if score > 0:
    #     predicted_label = 1
    # else:
    #     predicted_label = -1

    # # If the predicted label is different from the true label, update the weight vector
    # if predicted_label != truth:
    #     w = w + truth * email
    
    return w
