#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np

def naivebayesPY(x, y):
    X = np.array(x)
    Y = np.array(y).flatten()

    Y_smooth = np.hstack((Y, [1, -1]))

    pos_count = np.sum(Y_smooth == 1)
    neg_count = np.sum(Y_smooth == -1)
    total = len(Y_smooth)

    pos = pos_count / total
    neg = neg_count / total

    return pos, neg