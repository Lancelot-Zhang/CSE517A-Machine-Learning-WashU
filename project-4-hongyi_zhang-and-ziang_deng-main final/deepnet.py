# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:11:01 2019

@author: Jerry Xing, MN (updated Apr 4 2024)
"""
import numpy as np
from get_transition_func import get_transition_func
from forward_pass import forward_pass
from compute_loss import compute_loss
from backprop import backprop
def deepnet(Ws, xTr, yTr, wst, transname = 'sigmoid'):

    # Define start ids to pull out entries in Ws for each layer (using architechture wst)
    # Starting at 0 goint in REVERSE order of the network layers.
    num_W_par_per_layer = wst[0:-1] * wst[1:]
    num_b_par_per_layer = wst[0:-1]
    num_par_per_layer = num_W_par_per_layer + num_b_par_per_layer

    start_entry = np.hstack([0,np.cumsum(num_par_per_layer)])

    # Ws has not been initialized
    if Ws.size == 0:
        Ws = np.random.randn(start_entry[-1], 1) / 2

    # Create list of W matrices (where b is folded into W).
    # One W per layer starting with LAST layer.
    W = []
    for i in range(len(start_entry)-1):
        W.append(np.reshape(Ws[start_entry[i]:start_entry[i+1]], [wst[i], wst[i+1]+1]))

    trans_func, trans_func_der = get_transition_func(transname)

    aas, zzs = forward_pass(W, xTr, trans_func)

    if len(yTr) == 0:
        loss = zzs[0]
        return loss

    loss = compute_loss(zzs, yTr)
    gradient_list_of_matrices = backprop(W, aas,zzs, yTr, trans_func_der)

    # Reformat the gradient from a list of matrices to one vector (same format as Ws)
    gradient = np.zeros((start_entry[-1], 1))
    for i in range(len(start_entry)-1):
        gradient[start_entry[i]:start_entry[i+1]] = np.reshape(gradient_list_of_matrices[i], (start_entry[i+1] - start_entry[i], 1))

    return loss, gradient
