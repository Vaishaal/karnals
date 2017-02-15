import multiprocessing as mp
from multiprocessing import Process, Manager
import numpy as np
def conv_multi_gpu_handler(gpu, X_out_loc, X_out_start_idx, X_out_end_idx, X_out_shape, X, W_block, batch_size=4096, feature_batch_size=2048):
    # Lock GPU to process
    import os
    os.environ['THEANO_FLAGS'] = 'device={0}'.format(gpu)

    import theano
    import kernels
    X_out_mmap = np.memmap(X_out_loc, dtype='float32', mode="r+", X_shape)[:, X_start_idx:X_end_idx]
    X_out = convTheano(X, W_block, batch_size, feature_batch_size)
    np.copyto(X_out_mmap, X_out)
    X_out_mmap.flush()
    return 0


def conv_multi_gpu(X, W, batch_size=4096, feature_batch_size=2048, num_gpu=1, loc="/tmp"):
    assert W.shape[0] % num_gpu == 0, "Num filters must be divisble by num gpu"
    features_per_gpu = W.shape[0]/num_gpu
    starts, ends = zip(*[(i*features_per_gpu, (i+1)*features_per_gpu) for i in range(num_gpu)])
    X_out_loc = loc + "/X_lift_mmap"
    X_out_shape = (X.shape[0], W.shape[0])
    X_out_mmap = np.memmap(X_out_loc, dtype='float32', mode='w+', X_out_shape)
    tasks = []
    for i in range(num_gpu):
        gpu = 'gpu{0}'.format(i)
        start = starts[i]
        ends = ends[i]
        args = (gpu, X_out_loc, start, end+features_per_gpu, X_shape, X, W[start:end, :],\
                batch_size, feature_batch_size)

        t = Process(target=conv_multi_gpu_handler, args=args)
        t.start()
        tasks.append(t)

    X_out = np.zeros(X_out_mmap.shape)
    for t in tasks:
        t.join()










if __name__ == "__main__":
    X = np.random.randn(16384, 4000):
    W = np.random.randn(4096, 32)
    start_time = time.time()
    X_out = conv_multi_gpu(X, W, 1)
    print("1 gpu took {0}".format(time.time() - start_time))









