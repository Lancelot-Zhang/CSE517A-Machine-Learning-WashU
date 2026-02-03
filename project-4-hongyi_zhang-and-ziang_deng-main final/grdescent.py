# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:03:09 2019

@author: Jerry Xing, MN (updated Apr 4 2024)
"""
import numpy as np
def grdescent(func, w0, stepsize, maxiter, tolerance=1e-2):

    eps = np.finfo(float).eps
    w = w0.copy()
    iter_count = 0
    
    while iter_count < maxiter:
        f, g = func(w)
        
        if np.linalg.norm(g) < tolerance:
            break

        w = w - stepsize * g
        iter_count += 1
        
    return w
