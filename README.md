A Numba Cuda Implementation of KMeans Clustering

- PKMeans.py: The parallel implementation of KMeans

- EPKMeans.py: A parallel extension of PKMeans which allows for running multiple instances of kmeans at the same time (similar to SKLearn's n_items) (Still under construction)

- bmk.py: A 3D-points benchmark with clustering visualization

- mnist_bench.py: A benchmark comparing parallel KMeans to SKLearn's KMeans on the MNIST dataset

- mnist_clustering.py: A demonstration of parallel KMeans on the MNIST dataset

- KMeans.py: A naive implementation of KMeans in numpy


For calling parallel KMeans, refer to mnist_clustering.py or the documentation included in PKMeans.py

Specifically, you can use the following to invoke PKMeans:

from PKMeans import PKMeans

...

# data is a (number-of-samples, data-dimensionality) numpy matrix
# center is a reference centerpoint for generating random centroids
# k is your chosen k
# iters is number of iterations to perform

assignments, centroids = PKMeans(data, center, k=k, iters=iters)

# assignments will be a (number-of-samples) vector of cluster assignments
# centroids will be a (k, data-dimensionality) matrix of centroids
