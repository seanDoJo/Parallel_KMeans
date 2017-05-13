from PKMeans import PKMeans
from mnist import Numbers
from sklearn.cluster import KMeans
import numpy as np
import time

nums = Numbers()

limits = [i for i in range(500, 50000, 500)]

k=9

for limit in limits:

	data = np.matrix((256*nums.train_x[:limit]).astype(np.float64))

	t = time.time()
	aa, cll = PKMeans(data, 128, k=k, iters=100)
	pk_t = time.time() - t

	t = time.time()
	aa, cll = PKMeans(data, 128, k=k, iters=100, shared=False)
	sk_t = time.time() - t


	print "mnist,{},{},{},{}".format(limit, pk_t, sk_t, sk_t/pk_t)
