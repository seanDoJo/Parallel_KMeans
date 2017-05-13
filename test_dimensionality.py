import numpy as np
from PKMeans import PKMeans

import time

def generate(numpoints, ctr, std, dim):
    centers = np.random.uniform(0, ctr, size=3)
    stdevs = np.random.uniform(1, std, size=3)
    dt = []
    for i in range(dim):
    	xs = np.random.normal(np.random.uniform(0, ctr, size=1)[0], np.random.uniform(1, std, size=1)[0], (1, numpoints))
        dt.append(xs[0])

    return np.array(dt).T



# Number of clusters
KK = 5
dims = [i for i in range(2, 1024, 2)]
for dim in dims:

	d = None
	for i in range(KK):
		# How many points to generate per cluster
		di = generate(10000, (i+1)*10, 5, dim)
		if d is not None:
			d = np.concatenate((d, di))
		else:
			d = di


	d = np.matrix(d)

	start = time.time()
	ah,c = PKMeans(d, 100, k=KK, iters=100)
	pk_t = time.time() - start

	print "{},{}".format(dim, pk_t)
