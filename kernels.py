# conv kernels for comp bio experiments
from numba import jit
import csv
import math
import numpy as np

from theano import function, config, shared, sandbox
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d


def computeDistanceMatrix(XTest, XTrain):
    XTrain = XTrain.reshape(XTrain.shape[0], -1)
    XTest = XTest.reshape(XTest.shape[0], -1)
    XTrain_norms = (np.linalg.norm(XTrain, axis=1) ** 2)[:, np.newaxis]
    XTest_norms = (np.linalg.norm(XTest, axis=1) ** 2)[:, np.newaxis]
    K = XTest.dot(XTrain.T)
    K *= -2
    K += XTrain_norms.T
    K += XTest_norms
    return K

def computeRBFGramMatrix(XTest, XTrain, gamma=1):
    gamma = -1.0 * gamma
    print "Gamma is "+  str(gamma)
    return np.exp(gamma*computeDistanceMatrix(XTest, XTrain))

@jit(nopython=True)
def generateNgrams(x, n, alpha_size=4):
    outsize = int(x.shape[0]/alpha_size - n + 1)
    ngrams = np.zeros((outsize, n*alpha_size))
    for i in range(outsize):
        ngrams[i] = x[i*alpha_size:(i+n)*alpha_size]
    return ngrams

def generateConvFeatures(X, W, offset=None,  gpu=False, alpha_size=4):
    n = W.shape[-1]/4
    if (gpu):
        print("GPU IMPLEMENTATION STILL BUGGY")
        conv_out = convTheano(X,W)
    else:
        conv_out = convCPU(X,W, offset)

    # Pool
    pool_out = np.sum(conv_out, axis=1)

    return pool_out.reshape(pool_out.shape[0], pool_out.shape[-1])


def convCPU(X, W, offset, alpha_size=4):
    conv_out_shape = (X.shape[-1] - W.shape[-1])/4 + 1
    X_lift = np.zeros((X.shape[0], conv_out_shape, W.shape[0]))
    scale = np.sqrt(2/float(W.shape[0]))
    for i in range(X.shape[0]):
        if (i % 1000 == 0):
            print i, "Images Convolved"
        ngrams = generateNgrams(X[i,:], W.shape[-1]/4, alpha_size)
        xlift_conv = np.dot(ngrams, W.T)
        xlift_conv += offset
        np.cos(xlift_conv, xlift_conv)
        X_lift[i] = xlift_conv
    return X_lift

def convTheano(X, W, batch_size=4096, offset=0, alpha_size=4):
    X = X.reshape(X.shape[0], 1, 1, X.shape[1])
    W = W.reshape(W.shape[0], 1, 1, W.shape[1])
    d = W.shape[-1]
    D = W.shape[0]
    offsetTheano = shared(offset)
    conv_out_shape = (X.shape[-1] - W.shape[-1])/alpha_size + 1
    XOut = np.zeros((X.shape[0], conv_out_shape, W.shape[0]))
    num_batches = int(np.ceil(X.shape[0]/batch_size))
    WTheano = shared(W)
    for b in range(num_batches):
        print "Data Batch ", b
        end = min((b+1)*batch_size, X.shape[0])
        start = b*batch_size
        size = end - start
        XBatch = X[start:end]
        XBatchTheano = shared(XBatch)
        conv_out = conv2d(XBatchTheano, WTheano, subsample=(1, 4), filter_flip=False)
        conv_out = conv_out + offset
        conv_out = T.cos(conv_out)
        conv_out = conv_out.eval().reshape(size, D, conv_out_shape)
        XOut[start:end, :, :] = conv_out.transpose(0,2,1)
    return XOut



