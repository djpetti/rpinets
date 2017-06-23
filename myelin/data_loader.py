""" Manages downloading and storing the MNIST dataset. """


# This forks processes, so we want to import it as soon as possible, when there
# is as little memory as possible being used.
from . import cache, image_getter, imagenet

import cPickle as pickle
import json
import logging
import os
import random
import signal
import sys
import threading

import cv2

import numpy as np



logger = logging.getLogger(__name__)


class Loader(object):
  """ Generic superclass for anything that loads input data. """

  def __init__(self):
    # Handle to the actual buffers containing images.
    self._training_buffer = None
    self._testing_buffer = None
    self._training_labels = None
    self._testing_labels = None
    self._training_names = None
    self._testing_names = None

    self._train_set_size = None
    self._test_set_size = None
    self._valid_set_size = None

    self._training_buffer_mean = 0
    self._testing_buffer_mean = 0

  def get_train_set(self):
    """ Gets the next batch of training data. """
    raise NotImplementedError("This method must be overidden by a subclass.")

  def get_test_set(self):
    """ Gets the next set of testing data. """
    raise NotImplementedError("This method must be overidden by a subclass.")

  def get_train_batch_size(self):
    """ Returns: The size of the training batches. """
    return self._train_batch_size

  def get_test_batch_size(self):
    """ Returns: The size of the testing batches. """
    return self._test_batch_size


class DataManagerLoader(Loader):
  """ Loads datasets concurrently with the help of the data_manager package. """

  def __init__(self, batch_size, load_batches, image_shape, cache_location,
               dataset_location, patch_shape=None, patch_flip=True,
               link_with=[], pca_stddev=25, jitter_stddev=0,
               raw_labels=False):
    """
    Args:
      batch_size: How many images are in each batch.
      load_batches: How many batches to have in VRAM at any given time.
      image_shape: The shape of the images that will be loaded.
      cache_location: The location of the image cache.
      dataset_location: The common part of the path to the files that we will be
      loading our training and testing datasets from.
      patch_shape: The shape of the patches that will be extracted from the
      images. If None, no patches will be extracted, and the raw images will be
      used directly.
      patch_flip: Whether to include flipped patches.
      link_with: List of external cache directories to link with when we load
                 datasets.
      pca_stddev: The standard deviation for PCA.
      jitter_stddev: The standard deviation for jitter.
      raw_labels: If true, it returns raw labels instead of numerical mappings. """
    super(DataManagerLoader, self).__init__()

    self._image_shape = image_shape
    self._cache_location = cache_location
    self._dataset_location = dataset_location
    self._patch_shape = patch_shape
    self._patch_flip = patch_flip
    self.__link_with = link_with
    self.__raw_labels = raw_labels

    self._pca_stddev = pca_stddev
    self._jitter_stddev = jitter_stddev

    # Register signal handlers.
    signal.signal(signal.SIGTERM, self.__on_signal)
    signal.signal(signal.SIGINT, self.__on_signal)

    self._batch_size = batch_size
    self._load_batches = load_batches
    self._buffer_size = batch_size * load_batches
    logger.debug("Nominal buffer size: %d" % (self._buffer_size))

    self._train_batch_size = self._buffer_size
    if not self._patch_shape:
      # No patches.
      self._test_batch_size = self._buffer_size
    elif not patch_flip:
      # We only have five patches.
      self._test_batch_size = self._buffer_size * 5
    else:
      # We have to account for all the patches.
      self._test_batch_size = self._buffer_size * 10

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
    # Lock to protect accesses to data in CPU memory.
    self.__train_cpu_lock = threading.Lock()
    self.__test_cpu_lock = threading.Lock()

    self.__batch_size = batch_size
    self.__load_batches = load_batches

    # Force it to wait for data initially.
    self.__train_buffer_full.acquire()
    self.__test_buffer_full.acquire()

    # Labels have to be integers, so that means we have to map labels to
    # integers.
    self.__labels = {}
    # Map that goes in the opposite direction.
    self.__reverse_labels = {}
    self.__current_label = 0
    # Image ID values for the loaded images.
    self._training_names = []
    self._testing_names = []

    # This is an event that signals to the internal threads that it's time to
    # exit.
    self.__exit_event = threading.Event()

    # Start the loader threads.
    self._init_loader_threads()

    self.__cleaned_up = False

  def __del__(self):
    """ Cleanup upon program exit. """
    self.exit_gracefully()

  def __on_signal(self, *args, **kwargs):
    """ Upon receiving a signal, it cleans up and exits the program. """
    logger.error("Got signal, exiting.")

    self.exit_gracefully()
    sys.exit(1)

  def exit_gracefully(self):
    """ Stop the threads and exit properly. """
    if self.__cleaned_up:
      # We don't need to do this again.
      return

    logger.info("Data loader system is exiting NOW.")

    # Signal internal threads that it's time to quit.
    self.__exit_event.set()

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

    # Wait for the internal threads to join.
    self._join_loader_threads()
    # Cleanup the image getter.
    self._image_getter.cleanup()

    self.__cleaned_up = True

  def _init_image_getter(self):
    """ Initializes the specific ImageGetter that we will use to get images.
    This can be overriden by subclasses to add specific functionality. """
    # We don't want to load multiple batches for testing data, because this
    # generally requires a massive amount of memory, and provides minimal
    # performance benefits.
    batch_sizes = (self._buffer_size, self._batch_size)

    self._image_getter = \
        image_getter.ImageGetter(self._cache_location, batch_sizes,
                                 self._image_shape, preload_batches=2,
                                 load_datasets_from=self._dataset_location,
                                 patch_shape=self._patch_shape,
                                 patch_flip=self._patch_flip,
                                 link_with=self.__link_with,
                                 pca_stddev=self._pca_stddev,
                                 jitter_stddev=self._jitter_stddev)

  def _init_loader_threads(self):
    """ Starts the training and testing loader threads. """
    self._test_thread = threading.Thread(target=self._run_test_loader_thread)
    self._test_thread.start()

    self._train_thread = threading.Thread(target=self._run_train_loader_thread)
    self._train_thread.start()

  def _join_loader_threads(self):
    """ Joins the training and testing loader threads. If you override
    _init_loader_threads(), you should probably override this method too. """
    logger.info("Joining threads...")
    self._train_thread.join()
    self._test_thread.join()

  def _load_raw_training_batch(self):
    """ Loads raw image and label data from somewhere.
    This can be overriden by subclasses to add specific functionality.
    Returns:
      The loaded images, labels, and names. """
    return self._image_getter.get_random_train_batch()

  def _load_raw_testing_batch(self):
    """ Loads raw image and label data from somewhere.
    This can be overriden by subclasses to add specific functionality.
    Returns:
      The loaded images, labels, and names. """
    return self._image_getter.get_random_test_batch()

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
        self.__reverse_labels[self.__current_label] = label
        self.__current_label += 1

    return converted

  def __section_means(self, image_buffer):
    """ Calculates means for the section of the image buffer that came from each
    dataset in the case of linked datasets.
    Args:
      image_buffer: The image buffer to calculate the means of.
    Returns:
      The calculated means. """
    total_caches = len(self.__link_with) + 1
    section_size = image_buffer.shape[0] / total_caches
    logger.debug("Mean section size: %d" % (section_size))
    buffer_means = []

    for i in range(0, total_caches):
      start_index = i * section_size
      end_index = start_index + section_size
      section_mean = np.mean(image_buffer[start_index:end_index])
      section_mean = section_mean.astype("int16")

      buffer_means.append(section_mean)

    return buffer_means

  def __subtract_section_means(self, image_buffer, means):
    """ Subtracts the calculated section means from the buffer.
    Args:
      image_buffer: The buffer to subtract from.
      means: The calculated section means. """
    section_size = image_buffer.shape[0] / len(means)

    for i, mean in enumerate(means):
      start_index = i * section_size
      end_index = start_index + section_size

      image_buffer[start_index:end_index] -= mean

  def __load_next_training_batch(self):
    """ Loads the next batch of training data from the Imagenet backend. """
    self._training_buffer, labels, names = self._load_raw_training_batch()
    logger.debug("Got raw labels: %s" % (labels))

    # Compute separate means for each linked dataset.
    self._training_buffer_mean = self.__section_means(self._training_buffer)
    logger.debug("Training mean: %s" % (self._training_buffer_mean))

    self.__train_cpu_lock.acquire()

    self._training_names = names
    # Convert labels.
    if not self.__raw_labels:
      self._training_labels = self.__convert_labels_to_ints(labels)
    else:
      # Use the raw ones.
      self._training_labels = labels

    self.__train_cpu_lock.release()

  def __load_next_testing_batch(self):
    """ Loads the next batch of testing data from the Imagenet backend. """
    self._testing_buffer, labels, names = self._load_raw_testing_batch()
    logger.debug("Got raw labels: %s" % (labels))
    self._testing_buffer_mean = self.__section_means(self._testing_buffer)
    logger.debug("Testing mean: %s" % (self._testing_buffer_mean))

    self.__test_cpu_lock.acquire()

    self._testing_names = names
    # Convert labels.
    if not self.__raw_labels:
      self._testing_labels = self.__convert_labels_to_ints(labels)
    else:
      # Use the raw ones.
      self._testing_labels = labels

    self.__test_cpu_lock.release()

  def _run_train_loader_thread(self):
    """ The main function for the thread to load training data. """
    while True:
      # Make sure we don't write over our old batch.
      self.__train_buffer_empty.acquire()

      self.__image_getter_lock.acquire()
      logger.info("Loading next training batch from imagenet...")
      self.__load_next_training_batch()
      logger.info("Done loading next training batch.")

      thread_error = None
      try:
        self.__image_getter_lock.release()
        # Allow the main thread to use what we loaded.
        self.__train_buffer_full.release()
      except threading.ThreadError as e:
        # The only way this should happen is if we hit an exit condition.
        thread_error = e

      if self.__exit_event.is_set():
        logger.info("Got exit event, terminating train loader thread.")
        return

      if thread_error:
        raise thread_error

  def _run_test_loader_thread(self):
    """ The main function for the thread to load training data. """
    while True:
      # Make sure we don't write over our old batch.
      self.__test_buffer_empty.acquire()

      self.__image_getter_lock.acquire()
      logger.info("Loading next testing batch from imagenet...")
      self.__load_next_testing_batch()
      logger.info("Done loading next testing batch.")

      thread_error = None
      try:
        self.__image_getter_lock.release()
        # Allow the main thread to use what we loaded.
        self.__test_buffer_full.release()
      except threading.ThreadError as e:
        # The only way this should happen is if we hit an exit condition.
        thread_error = e

      if self.__exit_event.is_set():
        logger.info("Got exit event, terminating test loader thread.")
        return

      if thread_error:
        raise thread_error

  def get_train_set(self):
    """ Gets the next batch of training data. """
    logger.info("Waiting for new training data to be ready...")
    self.__train_buffer_full.acquire()
    logger.info("Got raw training data.")

    # Create a copy of the training data.
    training_buffer = self._training_buffer.astype("int16")
    self.__subtract_section_means(training_buffer, self._training_buffer_mean)
    labels = self._training_labels[:]
    # Allow it to load another batch.
    self.__train_buffer_empty.release()

    return (training_buffer, labels)

  def get_test_set(self):
    """ Gets the next batch of testing data. """
    logger.info("Waiting for new testing data to be ready...")
    self.__test_buffer_full.acquire()
    logger.info("Got raw testing data.")

    # Create a copy of the testing data.
    testing_buffer = self._testing_buffer.astype("int16")
    self.__subtract_section_means(testing_buffer, self._testing_buffer_mean)

    if (self._patch_shape and not self.__link_with):
      # Keras wants all extra copies.
      if self._patch_flip:
        labels = np.tile(self._testing_labels, 10)
      else:
        labels = np.tile(self._testing_labels, 5)
    else:
      labels = self._testing_labels[:]

    # Allow it to load another batch.
    self.__test_buffer_empty.release()

    return (testing_buffer, labels)

  def get_test_names(self):
    """
    Returns:
      A list of the image names of the loaded images for the test set. """
    self.__test_cpu_lock.acquire()
    names = self._testing_names[:]
    self.__test_cpu_lock.release()

    return names

  def get_train_names(self):
    """
    Returns:
      A list of the image names of the loaded images for the train set. """
    self.__train_cpu_lock.acquire()
    names = self._training_names[:]
    self.__train_cpu_lock.release()

    return names

  def get_train_set_size(self):
    """
    Returns:
      The total number of images in the training dataset. """
    return self._image_getter.get_train_set_size()

  def get_test_set_size(self):
    """
    Returns:
      The total number of images in the testing dataset. """
    return self._image_getter.get_test_set_size()

  def save(self, filename):
    """ Allows the saving of label associations for later use.
    Args:
      filename: The name of the file to write the saved data to. """
    file_object = open(filename, "wb")
    pickle.dump((self.__labels, self.__reverse_labels, self.__current_label),
                file_object)
    file_object.close()

  def load(self, filename):
    """ Loads label associations that have been saved to a file.
    Args:
      filename: The name of the file to load from. """
    file_object = file(filename, "rb")
    self.__labels, self.__reverse_labels, self.__current_label = \
        pickle.load(file_object)
    logger.debug("Starting at label %d." % (self.__current_label))

    file_object.close()

  def convert_ints_to_labels(self, output):
    """ Converts the output from a network tester or predictor to the actual
    corresponding labels.
    Args:
      output: A list of numbers to convert.
    Returns:
      A list of the actual labels. """
    labels = []

    self.__train_cpu_lock.acquire()
    self.__test_cpu_lock.acquire()

    for number in output:
      if number not in self.__reverse_labels:
        logger.warning("Couldn't find label %d." % (number))
        labels.append(None)
        continue
      labels.append(self.__reverse_labels[number])

    self.__train_cpu_lock.release()
    self.__test_cpu_lock.release()

    return labels

class SequentialDataManagerLoader(DataManagerLoader):
  """ Same as DataManagerLoader, but loads sequential batches instead of random
  batches. This is useful for doing things like performing validation. """

  def _load_raw_training_batch(self):
    """ Loads raw image and label data from somewhere.
    This can be overriden by subclasses to add specific functionality.
    Returns:
      The loaded images and labels. """
    return self._image_getter.get_sequential_train_batch()

  def _load_raw_testing_batch(self):
    """ Loads raw image and label data from somewhere.
    This can be overriden by subclasses to add specific functionality.
    Returns:
      The loaded images and labels. """
    return self._image_getter.get_sequential_test_batch()

class ImagenetLoader(DataManagerLoader):
  """ Loads data from imagenet. """

  def __init__(self, batch_size, load_batches, cache_location, dataset_location,
               synset_location, synset_file):
    """ See superclass documentation for this method.
    Additional Args:
      synset_location: Where to store downloaded synset data.
      synset_file: The file to load the synsets to use from. """
    self.__synset_location = synset_location
    self.__synset_file = synset_file

    super(ImagenetLoader, self).__init__(batch_size, load_batches,
                                         (256, 256, 3), cache_location,
                                         dataset_location,
                                         patch_shape=(224, 224))

  def _init_image_getter(self):
    """ Initializes the specific ImageGetter that we will use to get images.
    """
    # We don't want to load multiple batches for testing data, because this
    # generally requires a massive amount of memory, and provides minimal
    # performance benefits.
    batch_sizes = (self._buffer_size, self._batch_size)

    self._image_getter = \
        imagenet.SynsetFileImagenetGetter( \
            self.__synset_file, self.__synset_location, self._cache_location,
            batch_sizes, self._image_shape, preload_batches=2,
            load_datasets_from=self._dataset_location,
            patch_shape=self._patch_shape)

