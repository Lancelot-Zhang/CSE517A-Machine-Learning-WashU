# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:32:45 2019

@author: Jerry Xing, MN (updated Apr 4 2024)
"""
import numpy as np
def initweights(wst):
#
#    Returns a randomly initialized (flat) weight vector for a given neural network
#    architecture.
#
    num_par = 0
    for i in range(len(wst)-1):
        num_par = num_par + wst[i]*wst[i+1] + wst[i]

    W = np.random.randn(num_par, 1) / 2

    return W
