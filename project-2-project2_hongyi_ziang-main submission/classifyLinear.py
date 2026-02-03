#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np

def classifyLinear(x, w, b):

    X = np.asarray(x)
    W = np.asarray(w)
    
    scores = np.dot(W.T,X) + b
    
    preds = np.where(scores >= 0, 1, -1).astype(int)
    
    return preds
