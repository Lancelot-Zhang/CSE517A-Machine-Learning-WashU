from spamfilter import spamfilter
from trainspamfilter import trainspamfilter
from valsplit import valsplit

import numpy as np

# load the data:
data = np.load('data/data_train_default.npy', allow_pickle=True).item()
X = data['X']
Y = data['Y']

# split the data
# xTr and xVal will be of the shape d x n (num_dimensions x num_datapoints)
xTr,xVal,yTr,yVal = valsplit(X,Y)

# train spam filter with settings and parameters in trainspamfilter.py
w_trained = trainspamfilter(xTr,yTr)

# evaluate spam filter on validation set using default threshold
spamfilter(xVal,yVal,w_trained,0.1)
