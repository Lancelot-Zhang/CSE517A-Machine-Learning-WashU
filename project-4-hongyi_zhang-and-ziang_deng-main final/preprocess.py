# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:05:03 2019

@author: remus
"""
import numpy as np
def preprocess(xTr,xTe):

    d, n_train = np.shape(xTr)
    _, n_test  = np.shape(xTe)
    m = np.reshape(np.mean(xTr, 1), (d, 1))
    std = np.std(xTr, axis = 1)
    u = np.diag(1/std)
    xTr = u @ (xTr - m)
    xTe = u @ (xTe - m)
    
    return xTr, xTe, u, m