# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:03:45 2019

@author: remus
"""
import numpy as np
def compute_loss(zzs, yTr):
    
    delta = zzs[0] - yTr
    n = np.shape(yTr)[1]
    loss = np.sum(delta * delta) / (2 * n)
    
    return loss
