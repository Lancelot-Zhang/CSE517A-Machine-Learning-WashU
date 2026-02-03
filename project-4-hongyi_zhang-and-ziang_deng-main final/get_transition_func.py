# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:19:32 2019

@author: remus
Modified: Added Leaky ReLU and GELU activation functions
"""
import numpy as np

def get_transition_func(transType):
    """
    Given the type, gets a specific transition function
    
    INPUT
    transType: 'sigmoid', 'tanh', 'ReLU', 'leaky_relu', 'gelu'
    
    OUTPUT
    trans_func: transition function (function)
    trans_func_der: derivative of the transition function (function)
    """
    if transType.lower() == 'sigmoid':
        trans_func = lambda z: 1 / (1+np.exp(-z))
        trans_func_der = lambda z: trans_func(z)*(1-trans_func(z))
    elif transType.lower() == 'relu2':
        trans_func = lambda z: 0.5 * (np.maximum(z, 0)**2)
        trans_func_der = lambda z: (z>=0) * z
    elif transType.lower() == 'tanh':
        trans_func = lambda z: np.tanh(z)
        trans_func_der = lambda z: 1 - np.tanh(z)**2
    elif transType.lower() == 'relu':
        trans_func = lambda z: np.maximum(z, 0)
        trans_func_der = lambda z: z >= 0
    elif transType.lower() == 'leaky_relu':
        alpha = 0.01  # 泄漏系数
        trans_func = lambda z: np.maximum(alpha * z, z)
        trans_func_der = lambda z: np.where(z > 0, 1, alpha)
    elif transType.lower() == 'gelu':
        # 高斯误差线性单元激活函数 (GELU)
        trans_func = lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
        # GELU的近似导数
        trans_func_der = lambda z: 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3))) + \
                              0.5 * z * (1 - np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3))**2) * \
                              np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * z**2)
    else:
        raise ValueError('不支持的激活函数类型: ' + transType)
    
    return trans_func, trans_func_der