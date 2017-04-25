from numba import cuda, float32, float64
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import random


def initCentroids(k, ctr, std, dim):
    stdevs = np.random.uniform(0, std, size=dim)
    dt = []
    for j in range(dim): 
        xs = np.random.normal(
	    np.random.uniform(-ctr, ctr),
	    stdevs[j], 
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

    if globalId >= N:
        return

    minDist = -1.0
    ind = -1

    for cent_id in range(k):
	dist = 0.
	for dim in range(d):
	    shift = data[globalId*d + dim] - centroids[cent_id*(d+1) + dim]
	    dist += (shift * shift)
        if ind == -1 or dist < minDist:
            minDist = dist
            ind = cent_id

    assign[globalId] = ind

    cuda.syncthreads()

    #TODO: This is a race condition
    if globalId < k:
	tind = globalId * (d+1)
	for l in range((d+1)):
    		centroids[tind + l] = 0

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
	    #centroids[globalId*(d+1)+j] = n % (101)
	    centroids[globalId*(d+1)+j] = data[globalId*d+j]

def PKMeans(dta, ctr, k=8, iters=300, tpb=128, update_t=256, update_b=8):	

	N = dta.shape[0]
	dim = dta.shape[1]

	bpg = (N / tpb) + 1

	# Store the data as a 1-dimensional array
	dh = np.squeeze(np.asarray(dta.reshape((1, dta.shape[0]*dta.shape[1]))))

	# Centroid array stored as [[x, y, z, ... , number of points assigned to centroid], ...]
	ch = initCentroids(k, ctr, 0.25, dim)

	# Store the data cluster assignments as a 1-dimensional array
	ah = np.zeros(N, dtype=np.int32)

	dd = cuda.to_device(dh)
	cd = cuda.to_device(ch)
	ad = cuda.to_device(ah)

	avg_threads = k / 2
	for i in range(iters):
		closest[bpg, tpb](dd, ad, cd, N, k, dim)
		update_sum[update_b, update_t](dd, ad, cd, N, k, dim)
		update_avg[3, avg_threads](dd, ad, cd, N, k, dim)

	cuda.synchronize()

	ad.copy_to_host(ah)
	cd.copy_to_host(ch)

        return (ah, ch.reshape((k, (dim+1))))
