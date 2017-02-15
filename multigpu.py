import multiprocessing as mp
from multiprocessing import Process, Manager
import numpy as np
import time

def conv_multi_gpu_handler(gpu, X_out_loc, X_out_start_idx, X_out_end_idx, X_out_shape, X, W_block, batch_size=4096, feature_batch_size=1024):
    # Lock GPU to process
    import os
    gpu_int = int(gpu.replace("gpu",""))
    compile_dir ='base_compiledir=/home/ec2-user/.theano_{0}'.format(gpu_int)
    os.environ['THEANO_FLAGS'] = '{0},device={1}'.format(compile_dir,gpu)
    os.environ['CUDA_DEVICE'] = gpu.replace("gpu","")
    import theano
    import kernels
    import theano.misc.pycuda_init

    X_out_mmap = np.memmap(X_out_loc, dtype='float32', mode="r+", shape=X_out_shape)[:, X_out_start_idx:X_out_end_idx]
    print(gpu, W_block.shape[0], X_out_mmap.shape)

    feature_batch_size = min(feature_batch_size, W_block.shape[0])
    X_out = kernels.convTheano(X, W_block, batch_size, feature_batch_size, gpu=gpu)
    print("XOUT SHAPE", X_out.shape)
    np.copyto(X_out_mmap, X_out)
    X_out_mmap.flush()
    return 0


def conv_multi_gpu(X, W, batch_size=4096, feature_batch_size=1024, num_gpu=1, loc="/tmp"):
    assert W.shape[0] % num_gpu == 0, "Num filters must be divisble by num gpu"
    features_per_gpu = int(W.shape[0]/num_gpu)
    X_out_loc = loc + "/X_lift_mmap"
    X_out_shape = (X.shape[0], W.shape[0]*2)
    X_out_mmap = np.memmap(X_out_loc, dtype='float32', mode='w+', shape=X_out_shape)
    tasks = []
    for i in range(num_gpu):
        gpu = 'gpu{0}'.format(i)
        start_w = i*features_per_gpu
        end_w = (i+1)*features_per_gpu

        start_x = i*features_per_gpu*2
        end_x = (i+1)*features_per_gpu*2

        args = (gpu, X_out_loc, start_x, end_x, X_out_shape, X, W[start_w:end_w, :],\
                batch_size, feature_batch_size)

        t = Process(target=conv_multi_gpu_handler, args=args)
        t.start()
        tasks.append(t)

    X_out = np.zeros(X_out_mmap.shape)
    np.copyto(X_out, X_out_mmap)
    for t in tasks:
        t.join()
    return X_out










if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.randn(4096, 1200)
    W = np.random.randn(4096, 32)
    start_time = time.time()
    X_out = conv_multi_gpu(X, W, feature_batch_size=1024, batch_size=4096, num_gpu=4)
    print("4 gpu took {0}".format(time.time() - start_time))










