from PKMeans import PKMeans
from mnist import Numbers
from sklearn.cluster import KMeans
import numpy as np
import time

nums = Numbers()
iterss = [i for i in range(5, 500, 5)]
k=9
limit = 25000
for iters in iterss:

	data = np.matrix((256*nums.train_x[:limit]).astype(np.float64))

	t = time.time()
	aa, cll = PKMeans(data, 128, k=k, iters=iters)
	pk_t = time.time() - t


	print "mnist,{},{}".format(iters, pk_t)
