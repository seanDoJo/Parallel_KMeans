from numba import cuda, float32, float64
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import random
import time


def initCentroids(k, ctr, spread, dim):
    l = ctr - spread
    u = ctr + spread
    dt = []
    for j in range(dim): 
        xs = np.random.normal(
	    np.random.uniform(l, u),
	    0.1, 
	    (1, k)
        )
        dt.append(xs[0])
    dt.append(np.zeros((1, k))[0])

    d = np.matrix(dt).T
    flattened = np.squeeze(np.asarray(d.reshape((1, (dim+1)*k))))

    return flattened

@cuda.jit('void(float64[:], int32[:], float64[:], int32[:], uint64, uint64, uint64, uint64)')
def closest(data, assign, centroids, k, N, d, klen, w):

    globalId = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
    sid = cuda.threadIdx.x
    gid = globalId*d
    nt = cuda.blockDim.x

    s_centroids = cuda.shared.array(shape=0, dtype=float64)

    if globalId >= N:
        return

    for ki in range(klen):
        kk = k[ki]
        wl = w * ki
        minDist = -1.0
        ind = -1
        for cent_id in range(kk):
            cid = cent_id*(d+1)

            for idd in range(sid, d, nt):
                s_centroids[idd] = centroids[wl + cid + idd]
            cuda.syncthreads()
            dist = 0.
            for dim in range(d):
	        shift = data[gid + dim] - s_centroids[dim]
	        dist += (shift * shift)
            if ind == -1 or dist < minDist:
                minDist = dist
                ind = cent_id

        assign[N*ki + globalId] = ind

@cuda.jit('void(float64[:], int32[:], float64[:], int32[:], uint64, uint64, uint64, uint64)')
def zero(data, assign, centroids, k, N, d, klen, w):

    globalId = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
    nums = w*klen
    nt = cuda.blockDim.x * cuda.gridDim.x

    for i in range(globalId, nums, nt):
        centroids[i] = 0.

@cuda.jit('void(float64[:], int32[:], float64[:], int32[:], uint64, uint64, uint64, uint64)')
def update_sum(data, assign, centroids, k, N, d, klen, w):

    globalId = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
    numThreads = cuda.blockDim.x * cuda.gridDim.x
    
    for ki in range(klen):
        abase = N*ki
        cbase = w*ki
        for i in range(globalId, N, numThreads):
            # Sum up each dimension
            cent_id = assign[abase + i]
            for dim in range(d):
	        cuda.atomic.add(centroids, cbase + cent_id*(d+1) + dim, data[i*d + dim])

	    # Sum up the number of points belonging to each centroid 
            cuda.atomic.add(centroids, cbase + cent_id*(d+1) + d, 1)

@cuda.jit('void(float64[:], int32[:], float64[:], int32[:], uint64, uint64, uint64, uint64)')
def update_avg(data, assign, centroids, k, N, d, klen, w):

    globalId = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x

    for ki in range(klen):
        kk = k[ki]
        cbase = w*ki
        if globalId < kk:
            sf = centroids[cbase + globalId*(d+1) + d]
            if sf >= 1:
                for j in range(d):
	            centroids[cbase + globalId*(d+1) + j] /= sf

def EPKMeans(dta, ctr, k=None, iters=300, tpb=32, update_t=512, update_b=16):

    N = dta.shape[0]
    dim = dta.shape[1]

    if k is None:
        k = np.array([8])

    bpg = (N / tpb) + 1

    # Store the data as a 1-dimensional array
    dh = np.squeeze(np.asarray(dta.reshape((1, dta.shape[0]*dta.shape[1]))))

    # Store the data cluster assignments as a 1-dimensional array
    ah = np.zeros((N, k.shape[0]), dtype=np.int32)
    ah = np.squeeze(np.asarray(ah.reshape((1, N*k.shape[0]))))

    # Centroid array stored as [[x, y, z, ... , number of points assigned to centroid], ...]
    mk = k[np.argmax(k)]
    ch = []
    for ki in k:
        ch.append(initCentroids(mk, ctr, 1.5, dim))
    ch = np.array(ch)
    w = ch.shape[1]
    ch = ch.reshape((mk*(dim+1)*k.shape[0]))


    smsize = ch.dtype.itemsize * dim

    dd = cuda.to_device(dh)
    cd = cuda.to_device(ch)
    ad = cuda.to_device(ah)
    kd = cuda.to_device(k)

    avg_threads = mk / 2
    zero_blocks = (ch.shape[0] / 32) + 1


    for i in range(iters):
        closest[bpg, tpb, 0, smsize](dd, ad, cd, kd, N, dim, k.shape[0], w)

        zero[zero_blocks, 32, 0](dd, ad, cd, kd, N, dim, k.shape[0], w)

        update_sum[update_b, update_t, 0](dd, ad, cd, kd, N, dim, k.shape[0], w)

        update_avg[3, avg_threads, 0](dd, ad, cd, kd, N, dim, k.shape[0], w)


    cuda.synchronize()

    ad.copy_to_host(ah)
    cd.copy_to_host(ch)
    ch = ch.reshape((k.shape[0], mk, (dim+1)))
    ah = ah.reshape((k.shape[0], N))

    return (ah, ch)
