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

	sk_t_t = 0.
	pk_t_t = 0.

	for i in range(10):

		sk = KMeans(init='random', n_clusters=k, n_init=1)

		t = time.time()
		sk.fit(data)
		sk_t = time.time() - t
		sk_t_t += sk_t

		t = time.time()
		aa, cll = PKMeans(data, 128, k=k, iters=sk.n_iter_)
		pk_t = time.time() - t
		pk_t_t += pk_t

	pk_t = pk_t_t / 10.
	sk_t = sk_t_t / 10.

	print "{},{},{},{}".format(limit, sk_t, pk_t, sk_t / pk_t)
