""" Manages downloading and storing the MNIST dataset. """


import cPickle as pickle
import gzip
import os
import urllib2

import numpy as np
import theano
import theano.tensor as TT


MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
MNIST_FILE = "mnist.pkl.gz"


class Mnist(object):
  """ Deals with the MNIST dataset. """
  def __init__(self):
    self.__load()

  def __download_mnist(self):
    """ Downloads the mnist dataset from MNIST_URL. """
    print "Downloading MNIST data..."
    response = urllib2.urlopen(MNIST_URL)
    data = response.read()

    # Save it to a file.
    mnist_file = open(MNIST_FILE, "w")
    mnist_file.write(data)
    mnist_file.close()

  def __shared_dataset(self, data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.

    Args:
      data_xy: Pair of inputs and expected outputs for the dataset.
    Returns:
      Symbolic shared variable containing the dataset.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, TT.cast(shared_y, 'int32')

  def __load(self):
    """ Loads mnist dataset from the disk, or downloads it first if it isn't
    present.
    Returns:
      A training set, testing set, and a validation set. """
    if not os.path.exists(MNIST_FILE):
      # Download it first.
      self.__download_mnist()

    print "Loading MNIST from disk..."
    mnist_file = gzip.open(MNIST_FILE, "rb")
    train_set, test_set, valid_set = pickle.load(mnist_file)
    mnist_file.close()

    self.__train_set_size = train_set[1].shape[0]
    self.__test_set_size = test_set[1].shape[0]
    self.__valid_set_size = valid_set[1].shape[0]

    # Copy to shared variables.
    self.__shared_train_set = self.__shared_dataset(train_set)
    self.__shared_test_set = self.__shared_dataset(test_set)
    self.__shared_valid_set = self.__shared_dataset(valid_set)
    print "Done."

  def get_train_set(self):
    """ Returns: The training set. """
    return self.__shared_train_set

  def get_test_set(self):
    """ Returns: The testing set. """
    return self.__shared_test_set

  def get_valid_set(self):
    """ Returns: The validation set. """
    return self.__shared_valid_set

  def get_train_set_size(self):
    """ Returns: The size of the training set. """
    return self.__train_set_size

  def get_test_set_size(self):
    """ Returns: The size of the testing set. """
    return self.__test_set_size

  def get_valid_set_size(self):
    """ Returns: The size of the validation set. """
    return self.__valid_set_size
