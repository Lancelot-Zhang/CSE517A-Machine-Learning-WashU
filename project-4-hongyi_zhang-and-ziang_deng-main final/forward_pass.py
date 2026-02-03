# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:30:29 2019

@author: Jerry Xing, MN (updated Apr 4 2024)
"""
import numpy as np
def forward_pass(W, xTr, trans_func):
    n = np.shape(xTr)[1]
    zzs = [None] * (len(W) + 1)
    aas = [None] * (len(W) + 1)
    
    zzs[-1] = np.vstack((xTr, np.ones([1, n])))
    aas[-1] = xTr
    
    for i in range(len(W)-1, 0, -1):
        aas[i] = W[i] @ zzs[i+1]
        aas_activated = trans_func(aas[i])
        zzs[i] = np.vstack((aas_activated, np.ones([1, n])))
    
    aas[0] = W[0] @ zzs[1]
    zzs[0] = aas[0]
    
    return aas, zzs
