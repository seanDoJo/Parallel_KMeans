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

@cuda.jit('void(float64[:], int32[:], float64[:], uint64, uint64, uint64)')
def closest(data, assign, centroids, N, k, d):
    """
    data: The 1-dimensional array of datapoints (a flattened N x d matrix)

    assign: The 1-dimensional array of datapoint assignments (1 x N)

    centroids: The 1-dimensional array of centroid locations (a flattened k x (d+1) matrix)

    N: The number of datapoints

    k: The number of clusters

    d: The dimensionality of the points

    """ 

    globalId = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
    sid = cuda.threadIdx.x
    gid = globalId*d
    nt = cuda.blockDim.x

    s_centroids = cuda.shared.array(shape=0, dtype=float64)

    if globalId >= N:
        return

    minDist = -1.0
    ind = -1

    for cent_id in range(k):
	cid = cent_id*(d+1)

	for idd in range(sid, d, nt):
		s_centroids[idd] = centroids[cid + idd]
	cuda.syncthreads()
	dist = 0.
	for dim in range(d):
	    #shift = data[globalId*d + dim] - centroids[cid + dim]
	    shift = data[gid + dim] - s_centroids[dim]
	    dist += (shift * shift)
        if ind == -1 or dist < minDist:
            minDist = dist
            ind = cent_id

    assign[globalId] = ind

@cuda.jit('void(float64[:], int32[:], float64[:], uint64, uint64, uint64)')
def zero(data, assign, centroids, N, k, d):

    globalId = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x

    if globalId < k*(d+1):
	centroids[globalId] = 0.

@cuda.jit('void(float64[:], int32[:], float64[:], uint64, uint64, uint64)')
def update_sum(data, assign, centroids, N, k, d):
    """
    data: The 1-dimensional array of datapoints (a flattened N x d matrix)

    assign: The 1-dimensional array of datapoint assignments (1 x N)

    centroids: The 1-dimensional array of centroid locations (a flattened k x (d+1) matrix)

    N: The number of datapoints

    k: The number of clusters

    d: The dimensionality of the points

    """

    globalId = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
    numThreads = cuda.blockDim.x * cuda.gridDim.x
	
    for i in range(globalId, N, numThreads):
	# Sum up each dimension
        cent_id = assign[i]
	for dim in range(d):
	    cuda.atomic.add(centroids, cent_id*(d+1) + dim, data[i*d + dim])

	# Sum up the number of points belonging to each centroid 
	cuda.atomic.add(centroids, cent_id*(d+1) + d, 1)


@cuda.jit('void(float64[:], int32[:], float64[:], uint64, uint64, uint64)')
def update_avg(data, assign, centroids, N, k, d):

    globalId = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x

    if globalId >= k:
        return
	
    sf = centroids[globalId*(d+1) + d]

    #TODO: have each thread do this instead of a loop
    if sf >= 1:
	for j in range(d):
	    centroids[globalId*(d+1) + j] /= sf
    else:
	# Randomly initialize if no points belong to the centroid
	n = 3251.
	for j in range(d):
	    n = (n*n / 100) % 10000
	    centroids[globalId*(d+1)+j] = n % (10)
	    #centroids[globalId*(d+1)+j] = data[globalId*d+j]

def PKMeans(dta, ctr, k=8, iters=300, tpb=32, update_t=256, update_b=8):	

	N = dta.shape[0]
	dim = dta.shape[1]

	bpg = (N / tpb) + 1

	# Store the data as a 1-dimensional array
	dh = np.squeeze(np.asarray(dta.reshape((1, dta.shape[0]*dta.shape[1]))))

	# Centroid array stored as [[x, y, z, ... , number of points assigned to centroid], ...]
	ch = initCentroids(k, ctr, 1.5, dim)
	smsize = ch.dtype.itemsize * dim
	print smsize

	# Store the data cluster assignments as a 1-dimensional array
	ah = np.zeros(N, dtype=np.int32)

	t1 = time.time()
	dd = cuda.to_device(dh)
	cd = cuda.to_device(ch)
	ad = cuda.to_device(ah)
	t1 = (time.time() - t1)

	avg_threads = k / 2
	zero_blocks = (ch.shape[0] / 32) + 1

	ct = 0.
	zt = 0.
	ust = 0.
	uat = 0.

	for i in range(iters):
		#t = time.time()
		closest[bpg, tpb, 0, smsize](dd, ad, cd, N, k, dim)
		#cuda.synchronize()
		#ct += (time.time() - t)

		#t = time.time()
		zero[zero_blocks, 32, 0](dd, ad, cd, N, k, dim)
		#cuda.synchronize()
		#zt += (time.time() - t)

		#t = time.time()
		update_sum[update_b, update_t, 0](dd, ad, cd, N, k, dim)
		#cuda.synchronize()
		#ust += (time.time() - t)

		#t = time.time()
		update_avg[3, avg_threads, 0](dd, ad, cd, N, k, dim)
		#cuda.synchronize()
		#uat += (time.time() - t)

	cuda.synchronize()

	t2 = time.time()
	ad.copy_to_host(ah)
	cd.copy_to_host(ch)
	t2 = (time.time() - t2)

	print t1+t2

	#tt = float(ct + zt + ust + uat)

	#print "closest: {}, {}\nzero: {}, {}\nsum: {}, {}\navg: {}, {}\n".format((ct / tt), ct, (zt / tt), zt, (ust / tt), ust, (uat / tt), uat)

        return (ah, ch.reshape((k, (dim+1))))
