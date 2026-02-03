#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY

def naivebayes(x, y, x1):
    X = np.array(x)
    Y = np.array(y).flatten()
    x1 = np.array(x1).flatten()

    pos_prior, neg_prior = naivebayesPY(X, Y)
    pos_probs, neg_probs = naivebayesPXY(X, Y)

    pos_probs = pos_probs.flatten()
    neg_probs = neg_probs.flatten()

    eps = 1e-10
    log_pos = x1 * np.log(pos_probs + eps) + (1 - x1) * np.log(1 - pos_probs + eps)
    log_neg = x1 * np.log(neg_probs + eps) + (1 - x1) * np.log(1 - neg_probs + eps)

    log_likelihood_ratio = np.sum(log_pos - log_neg)
    log_prior_ratio = np.log(pos_prior / neg_prior)

    logratio = log_prior_ratio + log_likelihood_ratio
    print(logratio)

    return logratio