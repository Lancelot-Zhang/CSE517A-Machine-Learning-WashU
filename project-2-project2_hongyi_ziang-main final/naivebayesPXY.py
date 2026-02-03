#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
@author: Yichen
@author: M.Joo (smoothing with all zeros)
"""

import numpy as np

def naivebayesPXY(x, y):
    X = np.array(x)
    Y = np.array(y).flatten()

    d, n = X.shape

    X_smooth = np.hstack((X, np.ones((d, 2)), np.zeros((d, 2))))
    Y_smooth = np.hstack((Y, [-1, 1], [-1, 1]))

    pos_mask = (Y_smooth == 1)
    neg_mask = (Y_smooth == -1)
    pos_counts = X_smooth[:, pos_mask].sum(axis=1, keepdims=True)
    neg_counts = X_smooth[:, neg_mask].sum(axis=1, keepdims=True)

    m_pos = pos_mask.sum()
    m_neg = neg_mask.sum()

    posprob = pos_counts / m_pos
    negprob = neg_counts / m_neg

    return posprob, negprob

# =============================================================================
