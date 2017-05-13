from test_PKMeans import PKMeans
from mnist import Numbers
from sklearn.cluster import KMeans
import numpy as np
import time

nums = Numbers()

limits = [i for i in range(500, 50000, 500)]

k=9

for limit in limits:

	data = np.matrix((256*nums.train_x[:limit]).astype(np.float64))

	ct, zt, ust, uat, trt, tt = PKMeans(data, 128, k=k, iters=100)

	print "mnist,{},{},{},{},{},{},{}".format(limit, ct, zt, ust, uat, trt, tt)
