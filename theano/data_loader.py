""" Manages downloading and storing the MNIST dataset. """


# This forks processes, so we want to import it as soon as possible, when there
# is as little memory as possible being used.
from common.data_manager import cache, image_getter, imagenet

import cPickle as pickle
import gzip
import json
import logging
import os
import random
import urllib2
import signal
import threading

import cv2

import numpy as np

import theano
import theano.tensor as TT


MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
MNIST_FILE = "mnist.pkl.gz"

# Synset list for ILSVRC16.
ILSVRC16_SYNSETS = "/job_files/ilsvrc16_synsets.txt"
# Where to store downloaded synset data.
SYNSET_LOCATION = "/home/theano/training_data/synsets"
# Where to cache downloaded files.
CACHE_LOCATION = "/home/theano/training_data/cache"
# Where to write dataset information.
DATASET_LOCATION = "/home/theano/training_data/ilsvrc16_dataset"


logger = logging.getLogger(__name__)


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
    data_y = np.asarray(data_y)
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
    logger.info("Downloading MNIST data...")
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

    logger.info("Loading MNIST from disk...")
    mnist_file = gzip.open(MNIST_FILE, "rb")
    train_set, test_set, valid_set = pickle.load(mnist_file)
    mnist_file.close()

    # Reshape if we need to.
    if use_4d:
      logger.debug("Note: Using 4D tensor representation. ")

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


class DataManagerLoader(Loader):
  """ Loads datasets concurrently with the help of the data_manager package. """

  def __init__(self, batch_size, load_batches, dataset_location):
    """
    Args:
      batch_size: How many images are in each batch.
      load_batches: How many batches to have in VRAM at any given time.
      dataset_location: The common part of the path to the files that we will be
      loading our training and testing datasets from. """
    super(DataManagerLoader, self).__init__()

    self._dataset_location = dataset_location

    # Register signal handlers.
    signal.signal(signal.SIGTERM, self.__exit_gracefully)
    signal.signal(signal.SIGINT, self.__exit_gracefully)

    self._buffer_size = batch_size * load_batches
    # Handle to the actual buffers containing images.
    self.__training_buffer = None
    self.__testing_buffer = None
    self.__training_labels = None
    self.__testing_labels = None

    # This is how we'll actually get images.
    self._init_image_getter()
    # Lock that we use to make sure we are only getting one batch at a time.
    self.__image_getter_lock = threading.Lock()

    # These are used to signal the loader thread to load more data, and the main
    # thread to copy the loaded data.
    self.__train_buffer_empty = threading.Lock()
    self.__train_buffer_full = threading.Lock()
    self.__test_buffer_empty = threading.Lock()
    self.__test_buffer_full = threading.Lock()

    self.__batch_size = batch_size
    self.__load_batches = load_batches

    # Force it to wait for data initially.
    self.__train_buffer_full.acquire()
    self.__test_buffer_full.acquire()

    # Labels have to be integers, so that means we have to map labels to
    # integers.
    self.__labels = {}
    self.__current_label = 0

    # Start the loader threads.
    test_loader_thread = threading.Thread(target=self.__run_test_loader_thread)
    test_loader_thread.start()
    train_loader_thread = threading.Thread(target=self.__run_train_loader_thread)
    train_loader_thread.start()

  def _init_image_getter(self):
    """ Initializes the specific ImageGetter that we will use to get images.
    This can be overriden by subclasses to add specific functionality. """
    self._image_getter = \
        image_getter.ImageGetter(CACHE_LOCATION, self._buffer_size,
                                 preload_batches=2,
                                 load_datasets_from=self._dataset_location)

  def __exit_gracefully(self, *args, **kwargs):
    """ Exit properly when we get a signal. """
    logger.error("Got signal, exiting NOW.")

    # Release all the locks so nothing can be blocking on them.
    try:
      self.__train_buffer_empty.release()
    except threading.ThreadError:
      pass

    try:
      self.__train_buffer_full.release()
    except threading.ThreadError:
      pass

    try:
      self.__test_buffer_empty.release()
    except threading.ThreadError:
      pass

    try:
      self.__test_buffer_full.release()
    except threading.ThreadError:
      pass

    # Exit the program.
    sys.exit(1)

  def __convert_labels_to_ints(self, labels):
    """ Converts a set of labels from the default label names to integers, so
    that they can actually be used in the network.
    Args:
      labels: The labels to convert.
    Returns:
      A list of the converted labels. """
    converted = []
    for label in labels:
      if label in self.__labels:
        converted.append(self.__labels[label])
      else:
        # This is the first time we've seen this label.
        converted.append(self.__current_label)
        self.__labels[label] = self.__current_label
        self.__current_label += 1

    return converted

  def __load_next_training_batch(self):
    """ Loads the next batch of training data from the Imagenet backend. """
    self.__training_buffer, labels = \
        self._image_getter.get_random_train_batch()
    logger.debug("Got raw labels: %s" % (labels))
    mean = np.mean(self.__training_buffer)
    self.__training_buffer -= mean
    logger.debug("Training mean: %f" % mean)

    # Convert labels.
    self.__training_labels = self.__convert_labels_to_ints(labels)

    # Show image thumbnails.
    #film_strip = np.concatenate([np.transpose(i, (1, 2, 0)) \
    #                             for i in self.__training_buffer[0:8]], axis=1)
    #cv2.imshow("Input Images", film_strip)
    # Force it to update the window.
    #cv2.waitKey(1)

    self.__training_buffer = self.__training_buffer.astype(theano.config.floatX)
    # Standard AlexNet procedure is to subtract the mean.
    self.__training_buffer -= mean

  def __load_next_testing_batch(self):
    """ Loads the next batch of testing data from the Imagenet backend. """
    self.__testing_buffer, labels = \
        self._image_getter.get_random_test_batch()
    logger.debug("Got raw labels: %s" % (labels))
    mean = np.mean(self.__testing_buffer)
    logger.debug("Testing mean: %f" % mean)

    # Convert labels.
    self.__testing_labels = self.__convert_labels_to_ints(labels)

    # Show image thumbnails.
    #film_strip = np.concatenate([np.transpose(i, (1, 2, 0)) \
    #                             for i in self.__testing_buffer[0:8]], axis=1)
    #cv2.imshow("Input Images", film_strip)
    # Force it to update the window.
    #cv2.waitKey(1)

    self.__testing_buffer = self.__testing_buffer.astype(theano.config.floatX)
    # Standard AlexNet procedure is to subtract the mean.
    self.__testing_buffer -= mean

  def __run_train_loader_thread(self):
    """ The main function for the thread to load training data. """
    while True:
      # Make sure we don't write over our old batch.
      self.__train_buffer_empty.acquire()

      self.__image_getter_lock.acquire()
      logger.info("Loading next training batch from imagenet...")
      self.__load_next_training_batch()
      logger.info("Done loading next training batch.")

      self.__image_getter_lock.release()
      # Allow the main thread to use what we loaded.
      self.__train_buffer_full.release()

  def __run_test_loader_thread(self):
    """ The main function for the thread to load training data. """
    while True:
      # Make sure we don't write over our old batch.
      self.__test_buffer_empty.acquire()

      self.__image_getter_lock.acquire()
      logger.info("Loading next testing batch from imagenet...")
      self.__load_next_testing_batch()
      logger.info("Done loading next testing batch.")

      self.__image_getter_lock.release()
      # Allow the main thread to use what we loaded.
      self.__test_buffer_full.release()

  def __swap_in_training_data(self):
    """ Takes training data buffered into system memory and loads it into VRAM
    for immediate use. """
    logger.info("Waiting for new training data to be ready...")
    self.__train_buffer_full.acquire()
    logger.info("Loading new training dataset into VRAM...")

    self._shared_dataset((self.__training_buffer, self.__training_labels),
                         self._shared_train_set)

    # Allow it to load another batch.
    self.__train_buffer_empty.release()

  def __swap_in_testing_data(self):
    """ Takes testing data buffered into system memory and loads it into VRAM
    for immediate use. """
    logger.info("Waiting for new testing data to be ready...")
    self.__test_buffer_full.acquire()
    logger.info("Loading new testing data into VRAM...")

    self._shared_dataset((self.__testing_buffer, self.__testing_labels),
                         self._shared_test_set)

    # Allow it to load another batch.
    self.__test_buffer_empty.release()

  def get_train_set(self):
    # Load a new set for it.
    self.__swap_in_training_data()

    return super(Ilsvrc12, self).get_train_set()

  def get_test_set(self):
    # Load a new set for it.
    self.__swap_in_testing_data()

    return super(Ilsvrc12, self).get_test_set()

  def get_non_shared_test_set(self):
    """ Gets a non-shared version of the test set, useful for AlexNet. """
    return self.__testing_labels

  def save(self, filename):
    """ Allows the saving of label associations for later use.
    Args:
      filename: The name of the file to write the saved data to. """
    file_object = open(filename, "wb")
    pickle.dump(self.__labels, file_object)
    file_object.close()

  def load(self, filename):
    """ Loads label associations that have been saved to a file.
    Args:
      filename: The name of the file to load from. """
    file_object = file(filename, "rb")
    self.__labels = pickle.load(file_object)
    file_object.close()


class ImagenetLoader(DataManagerLoader):
  """ Loads data from imagenet. """

  def __init__(self, batch_size, load_batches):
    """ See superclass documentation for this method. """
    super(ImagenetLoader, self).__init__(batch_size, load_batches,
                                         DATASET_LOCATION)

  def _init_image_getter(self):
    """ Initializes the specific ImageGetter that we will use to get images.
    """
    self._image_getter = \
        imagenet.SynsetFileImagenetGetter( \
            ILSVRC16_SYNSETS, SYNSET_LOCATION, CACHE_LOCATION,
            self._buffer_size, preload_batches=2,
            load_datasets_from=self._dataset_location)

