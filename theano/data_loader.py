""" Manages downloading and storing the MNIST dataset. """


import cPickle as pickle
import gzip
import os
import random
import urllib2

import cv2

import numpy as np

import theano
import theano.tensor as TT

from common.imagenet_server import cache


MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
MNIST_FILE = "mnist.pkl.gz"

ILSVRC12_LOCATION = "/home/daniel/datasets/ilsvrc12"


class Loader(object):
  """ Generic superclass for anything that loads input data. """

  def __init__(self):
    self._shared_train_set = [None, None]
    self._shared_test_set = [None, None]
    self._shared_valid_set = [None, None]

    self._train_set_size = None
    self._test_set_size = None
    self._valid_set_size = None

  def _shared_dataset(self, data, shared_set):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.

    Args:
      data: The data to load.
      shared_set: The shared variables to load it into.
    Returns:
      Symbolic shared variable containing the dataset.
    """
    data_x, data_y = data
    if shared_set == [None, None]:
      # The shared variables weren't initialized yet.
      shared_set[0] = theano.shared(data_x.astype(theano.config.floatX))
      shared_set[1] = theano.shared(data_y.astype(theano.config.floatX))
    else:
      # They are initialized, we just need to set new values.
      shared_set[0].set_value(data_x.astype(theano.config.floatX))
      shared_set[1].set_value(data_y.astype(theano.config.floatX))

  def __cast_dataset(self, dataset):
    """ To store it on the GPU, it needs to be of type float32, however, the
    labels need to be type int, so we use this little casting hack.
    Args:
      dataset: The dataset to operate on.
    Returns:
      A version of dataset with the labels casted. """
    images, labels = dataset
    return (images, TT.cast(labels, "int32"))

  def get_train_set(self):
    """ Returns: The training set. """
    return self.__cast_dataset(self._shared_train_set)

  def get_test_set(self):
    """ Returns: The testing set. """
    return self.__cast_dataset(self._shared_test_set)

  def get_valid_set(self):
    """ Returns: The validation set. """
    return self.__cast_dataset(self._shared_valid_set)

  def get_train_set_size(self):
    """ Returns: The size of the training set. """
    return self._train_set_size

  def get_test_set_size(self):
    """ Returns: The size of the testing set. """
    return self._test_set_size

  def get_valid_set_size(self):
    """ Returns: The size of the validation set. """
    return self._valid_set_size


class Mnist(Loader):
  """ Deals with the MNIST dataset.
  Args:
    use_4d: If True, it will reshape the inputs to 4D tensors for use in a CNN.
            Defaults to False. """
  def __init__(self, use_4d=False):
    super(Mnist, self).__init__()

    self.__load(use_4d)

  def __download_mnist(self):
    """ Downloads the mnist dataset from MNIST_URL. """
    print "Downloading MNIST data..."
    response = urllib2.urlopen(MNIST_URL)
    data = response.read()

    # Save it to a file.
    mnist_file = open(MNIST_FILE, "w")
    mnist_file.write(data)
    mnist_file.close()

  def __load(self, use_4d):
    """ Loads mnist dataset from the disk, or downloads it first if it isn't
    present.
    Args:
      use_4d: If True, it will reshape the inputs to a 4D tensor for use in a
              CNN.
    Returns:
      A training set, testing set, and a validation set. """
    if not os.path.exists(MNIST_FILE):
      # Download it first.
      self.__download_mnist()

    print "Loading MNIST from disk..."
    mnist_file = gzip.open(MNIST_FILE, "rb")
    train_set, test_set, valid_set = pickle.load(mnist_file)
    mnist_file.close()

    # Reshape if we need to.
    if use_4d:
      print "Note: Using 4D tensor representation. "

      train_x, train_y = train_set
      test_x, test_y = test_set
      valid_x, valid_y = valid_set

      train_x = train_x.reshape(-1, 1, 28, 28)
      test_x = test_x.reshape(-1, 1, 28, 28)
      valid_x = valid_x.reshape(-1, 1, 28, 28)

      train_set = (train_x, train_y)
      test_set = (test_x, test_y)
      valid_set = (valid_x, valid_y)

    self._train_set_size = train_set[1].shape[0]
    self._test_set_size = test_set[1].shape[0]
    self._valid_set_size = valid_set[1].shape[0]

    # Copy to shared variables.
    self._shared_dataset(train_set, self._shared_train_set)
    self._shared_dataset(test_set, self._shared_test_set)
    self._shared_dataset(valid_set, self._shared_valid_set)
    print "Done."

class Ilsvrc12(Loader):
  """ Loads ILSVRC12 data that's saved to the disk. """
  def __init__(self, batch_size, load_batches, use_4d=False):
    """
    Args:
      load_batches: How many batches to have in VRAM at any given time.
      batch_size: How many images are in each batch.
      use_4d: If True, will reshape the inputs for use in a CNN. Defaults to
              False. """
    super(Ilsvrc12, self).__init__()

    self.__use_4d = use_4d

    self.__batch_size = batch_size
    self.__load_batches = load_batches
    self.__buffer = cache.MemoryBuffer(256,
                                       self.__batch_size * self.__load_batches,
                                       color=True)

    # Labels have to be integers, so that means we have to map synsets to
    # integers.
    self.__synets = {}
    self.__current_label = 0

  def __load_random_image(self):
    """ Loads a random image from the dataset.
    Returns:
      The image, and the synset that the image belongs to. """
    base_path = os.path.join(ILSVRC12_LOCATION, "train")

    while True:
      # Choose a synset.
      possible_synsets = os.listdir(base_path)
      synset = possible_synsets[random.randint(0, len(possible_synsets) - 1)]

      # Choose an image.
      synset_path = os.path.join(base_path, synset)
      possible_images = os.listdir(synset_path)
      image = possible_images[random.randint(0, len(possible_images) - 1)]

      # Open the image.
      image_path = os.path.join(synset_path, image)
      image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

      if image == None:
        print "WARNING: Failed to load image: %s" % (image_path)
        continue

      return (image, synset)

  def __load_new_set(self):
    """ Loads images from the disk into memory.
    Returns:
      The images from the new set, and the labels. """
    print "Loading new batches..."
    self.__buffer.clear()

    # Load data from disk.
    labels = []
    for i in range(0, self.__batch_size * self.__load_batches):
      image, synset = self.__load_random_image()
      self.__buffer.add(image, i)

      # Find a label.
      if synset in self.__synets:
        label = self.__synets[synset]
      else:
        # We need a new label.
        label = self.__current_label
        self.__synets[synset] = label
        self.__current_label += 1
      labels.append(label)

    self._train_set_size = self.__batch_size * self.__load_batches
    self._test_set_size = self.__batch_size * self.__load_batches
    self._valid_set_size = 0

    print "Done."

    images = self.__buffer.get_storage()
    # Reshape the images if need be.
    if self.__use_4d:
       images = images.reshape(-1, 3, 256, 256)

    return (images, np.asarray(labels))

  def get_train_set(self):
    # Load a new set for it.
    dataset = self.__load_new_set()
    self._shared_dataset(dataset, self._shared_train_set)

    return super(Ilsvrc12, self).get_train_set()

  def get_test_set(self):
    dataset = self.__load_new_set()
    self._shared_dataset(dataset, self._shared_test_set)

    return super(Ilsvrc12, self).get_test_set()
