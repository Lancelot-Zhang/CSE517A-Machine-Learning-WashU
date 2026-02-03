#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
@author: MN (categorical/Bernoulli NB)
"""

import numpy as np
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY

def naivebayesCL(x, y):
    X = np.array(x)
    Y = np.array(y).flatten()

    pos_prior, neg_prior = naivebayesPY(X, Y)
    pos_probs, neg_probs = naivebayesPXY(X, Y)

    eps = 1e-10
    pos_probs = np.clip(pos_probs, eps, 1-eps)
    neg_probs = np.clip(neg_probs, eps, 1-eps)

    w = np.log(pos_probs * (1 - neg_probs)) -np.log((neg_probs * (1 - pos_probs)))

    log_prior_ratio = np.log(pos_prior / neg_prior)
    log_feature_ratio = np.sum(np.log((1 - pos_probs) / (1 - neg_probs)))
    b = log_prior_ratio + log_feature_ratio

    return w, b.item()