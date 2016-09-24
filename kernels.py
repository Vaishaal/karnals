# conv kernels for comp bio experiments
from numba import jit
import csv
import math
import numpy as np

from theano import function, config, shared, sandbox
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
from theano.misc.pycuda_utils import *
from theano.gof.utils import give_variables_names
from pycuda.cumath import cos as pycuda_cos
from pycuda.compiler import SourceModule
from theano.gpuarray import dnn 
import theano.sandbox.cuda as cuda_ndarray
import theano

from theano.sandbox.cuda.var import CudaNdarrayVariable
from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_from_host

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
    print("Gamma is "+  str(gamma))
    return np.exp(gamma*computeDistanceMatrix(XTest, XTrain))

@jit(nopython=True)
def generateNgrams(x, n, alpha_size=4):
    outsize = int(x.shape[0]/alpha_size - n + 1)
    ngrams = np.zeros((outsize, int(n*alpha_size)))
    for i in range(outsize):
        ngrams[i] = x[i*alpha_size:(i+n)*alpha_size]
    return ngrams

def generateConvFeatures(X, W, offset=None, gpu=False, num_feature_batches=1, batch_size=4096, alpha_size=4):
    n = W.shape[-1]/4
    if (gpu):
        print("GPU IMPLEMENTATION STILL BUGGY")
        conv_out = convTheano(X,W,num_feature_batches=num_feature_batches,batch_size=batch_size)
    else:
        conv_out = convCPU(X,W, offset)

    return conv_out


def convCPU(X, W, offset, alpha_size=4):
    conv_out_shape = (X.shape[-1] - W.shape[-1])/4 + 1
    X_lift = np.zeros((X.shape[0], W.shape[0]))
    scale = np.sqrt(2/float(W.shape[0]))
    for i in range(X.shape[0]):
        if (i % 1000 == 0):
            print(i, "Images Convolved")
        ngrams = generateNgrams(X[i,:], W.shape[-1]/4, alpha_size)
        xlift_conv = np.dot(ngrams, W.T)
        xlift_conv += offset
        np.cos(xlift_conv, xlift_conv)
        pool_out = np.sum(xlift_conv, axis=0)
        X_lift[i] = pool_out

    return X_lift


def convTheano(X, W, batch_size=4096, num_feature_batches=1, feature_batch_size=2048, alpha_size=4):
    X = X.reshape(X.shape[0], 1, 1, X.shape[1]).astype('float32')
    W = W.reshape(W.shape[0], 1, 1, W.shape[1]).astype('float32')
    fbs = feature_batch_size
    d = W.shape[-1]
    D = W.shape[0]
    conv_out_shape = int((X.shape[-1] - W.shape[-1])/alpha_size + 1)
    XOut = np.zeros((X.shape[0], 2*W.shape[0]))
    num_batches = int(np.ceil(X.shape[0]/batch_size))
    num_feature_batches = int(np.ceil(W.shape[0]/feature_batch_size))
    WTheano = None
    XBatchTheano = None
    for fb in range(num_feature_batches):
        print("Feature Batch ", fb)
        f_start = fb*(fbs*2)
        f_end = (fb+1)*(fbs*2)
        W_block = W[fb*fbs:(fb+1)*fbs]
        if (WTheano == None):
            WTheano = shared(W_block)
        else:
            WTheano.set_value(W_block)

        for b in range(num_batches):
            print("Data Batch ", b)
            end = min((b+1)*batch_size, X.shape[0])
            start = b*batch_size
            size = end - start
            XBatch = X[start:end]

            # Set XBAtch
            if (XBatchTheano == None):
                XBatchTheano = shared(XBatch)
            else:
                XBatchTheano.set_value(XBatch)

            conv_out = conv2d(XBatchTheano, WTheano, subsample=(1, 4), filter_flip=False)

            # Fuck broadcasting I have to do this sin + cosine shit
            cos_out = cuda_cos(conv_out)
            pool_cos_out = pool_2d(cos_out, (1, conv_out_shape), mode='sum', ignore_border=False).eval()

            sin_out = cuda_sin(conv_out)
            pool_sin_out = pool_2d(sin_out, (1, conv_out_shape), mode='sum', ignore_border=False).eval()

            out = np.concatenate((pool_sin_out, pool_cos_out), axis=-1)
            XOut[start:end, f_start:f_end] = out.reshape(size, fbs*2)
    XBatchTheano.set_value([[[[]]]])
    WTheano.set_value([[[[]]]])
    return XOut

def cpu_to_gpu_var(x):
    type = cuda.CudaNdarrayType(broadcastable=x.broadcastable)
    name = gpu_name(x.name)
    gpu_var = cuda.CudaNdarrayVariable(type=type, name=name)
    cpu_var = cuda.host_from_gpu(gpu_var)
    return gpu_var, cpu_var



class CUDA_cos(theano.Op):
    __props__ = ()
    def make_node(self, inp):
        inp = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp))
        assert inp.dtype == "float32"
        return theano.Apply(self, [inp], [inp.type()])

    def make_thunk(self, node, storage_map, _, _2):
        mod = SourceModule("""
    __global__ void my_fct(float * i0, float * o0, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size){
        o0[i] = cosf(i0[i]);
    }
  }""")
        pycuda_fct = mod.get_function("my_fct")
        inputs = [ storage_map[v] for v in node.inputs]
        outputs = [ storage_map[v] for v in node.outputs]
        def thunk():
            z = outputs[0]
            if z[0] is None or z[0].shape!=inputs[0][0].shape:
                z[0] = cuda.CudaNdarray.zeros(inputs[0][0].shape)
            grid = (int(np.ceil(inputs[0][0].size / 512.)),1)
            pycuda_fct(inputs[0][0], z[0], np.intc(inputs[0][0].size),
                       block=(512, 1, 1), grid=grid)
        thunk.lazy = False
        return thunk

cuda_cos = CUDA_cos()


class CUDA_sin(theano.Op):
    __props__ = ()
    def make_node(self, inp):
        inp = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp))
        assert inp.dtype == "float32"
        return theano.Apply(self, [inp], [inp.type()])

    def make_thunk(self, node, storage_map, _, _2):
        mod = SourceModule("""
    __global__ void my_fct(float * i0, float * o0, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size){
        o0[i] = sinf(i0[i]);
    }
  }""")
        pycuda_fct = mod.get_function("my_fct")
        inputs = [ storage_map[v] for v in node.inputs]
        outputs = [ storage_map[v] for v in node.outputs]
        def thunk():
            z = outputs[0]
            if z[0] is None or z[0].shape!=inputs[0][0].shape:
                z[0] = cuda.CudaNdarray.zeros(inputs[0][0].shape)
            grid = (int(np.ceil(inputs[0][0].size / 512.)),1)
            pycuda_fct(inputs[0][0], z[0], np.intc(inputs[0][0].size),
                       block=(512, 1, 1), grid=grid)
        thunk.lazy = False
        return thunk

cuda_sin = CUDA_sin()

