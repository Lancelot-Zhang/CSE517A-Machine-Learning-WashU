# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:53:00 2019

@author: Jerry Xing, MN (updated Apr 4 2024)
"""
import numpy as np
import scipy.io as sio
from preprocess import preprocess
from initweights import initweights
from grdescent import grdescent
import matplotlib.pyplot as plt
from deepnet import deepnet
def bostondemo(wst = None):

    TRANSNAME='sigmoid'
    ROUNDS=200
    ITER=50
    STEPSIZE=0.01

    xTr = np.loadtxt("data/xTr.csv", delimiter=",")
    yTr = np.loadtxt("data/yTr.csv", delimiter=",").reshape(1,-1) #to make it (1,n)
    xTe = np.loadtxt("data/xTe.csv", delimiter=",")
    yTe = np.loadtxt("data/yTe.csv", delimiter=",").reshape(1,-1) #to make it (1,n)

    # Create your NN architecture:
    # Each element in wst is the number of nodes in that layer from BACK to FRONT. You
    # can also change the number of layers. The network must start with 1 and end with d.
    # Default for boston data: [1 20 20 20 13].
    if wst == None:
        wst = np.array([1,20,20,13,np.shape(xTr)[0]])

    [xTr,xTe, _, _] = preprocess(xTr,xTe)

    # Initalize the weights. This is a 1d array, one entry per parameter.
    # Use wst to assign parameters to layers and units.
    w = initweights(wst)

    # Sort X and y (for visualization)
    itr = np.argsort(yTr).flatten()
    ite = np.argsort(yTe).flatten()
    xTr = xTr[:, itr]
    xTe = xTe[:, ite]
    yTr = yTr[:, itr]
    yTe = yTe[:, ite]

    # Create dummy xtr and xte (for visualization)
    xtr = np.arange(0,np.shape(yTr)[1])
    xte = np.arange(0,np.shape(yTe)[1])

    plt.figure()
    plt.subplot(1,3,1);plt.title('TRAIN')
    plt.plot(xtr, yTr.flatten(), 'r', linewidth=5)
    linePredTr = plt.plot(xtr, np.ones(len(xtr)), 'k.', ms=3)
    plt.subplot(1,3,2);plt.title('TEST')
    plt.plot(xte,yTe.flatten(), 'r', linewidth=5)
    linePredTe = plt.plot(xte, np.ones(len(xte)), 'k.', ms=3)

    errTr = []
    errTe = []
    f = lambda w: deepnet(w, xTr, yTr, wst, TRANSNAME)
    for i in range(ROUNDS):
        w = grdescent(f, w, STEPSIZE, ITER, 1e-8)
        predTr=deepnet(w,xTr,[],wst,TRANSNAME)
        predTe=deepnet(w,xTe,[],wst,TRANSNAME)
        errTr.append(np.sqrt(np.mean((predTr-yTr)**2)))
        errTe.append(np.sqrt(np.mean((predTe-yTe)**2)))
        linePredTr[0].set_data(xtr, predTr.flatten())
        linePredTe[0].set_data(xte, predTe.flatten())
        plt.subplot(1,3,3);plt.title('RMSE Errors')
        lineErrTr = plt.plot(errTr,'b', label = 'train')
        lineErrTe = plt.plot(errTe,'r', label = 'test')
        plt.legend(handles=[lineErrTr[0], lineErrTe[0]])
        plt.pause(0.05)

    plt.show()
    print('Lowest train score: {}'.format(np.min(errTr)))
    print('Lowest test score: {}'.format(np.min(errTe)))

if __name__ == '__main__':
    bostondemo()
