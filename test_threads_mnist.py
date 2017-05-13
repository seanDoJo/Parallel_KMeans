from PKMeans import PKMeans
from mnist import Numbers
from sklearn.cluster import KMeans
import numpy as np
import time

nums = Numbers()
iters = 100
k=9
limit = 50000
threads = [16, 32, 64, 128, 256, 512]
for ti in threads:
	data = np.matrix((256*nums.train_x[:limit]).astype(np.float64))

	t = time.time()
	aa, cll = PKMeans(data, 128, k=k, iters=iters, tpb=ti)
	pk_t = time.time() - t


	print "mnist,{},{}".format(ti, pk_t)
