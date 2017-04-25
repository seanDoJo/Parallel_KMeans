import argparse
from collections import Counter, defaultdict

import random
import numpy
from numpy import median

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open("mnist.pkl.gz", 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()
