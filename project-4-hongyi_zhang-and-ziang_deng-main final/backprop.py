# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:07:53 2019

@author: Jerry Xing, MN (updated Apr 4 2024)
"""
import numpy as np
def backprop(W, aas, zzs, yTr, trans_func_der):
    n = np.shape(yTr)[1]
    delta = zzs[0] - yTr
    gradient = [None] * len(W)    
    gradient[0] = delta @ zzs[1].T / n

    for i in range(1, len(W)):

        delta = (W[i-1].T @ delta)[:-1, :]
        delta = delta * trans_func_der(aas[i])
        gradient[i] = delta @ zzs[i+1].T / n
    
    return gradient
