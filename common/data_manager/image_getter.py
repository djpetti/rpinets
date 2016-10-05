import logging

import cache
import dataset


logger = logging.getLogger(__name__)


class ImageGetter(object):
  """ Gets random sets of images for use in training and testing. """

  def __init__(self, cache_location, batch_size, preload_batches=1,
               test_percentage=0.1, load_datasets_from=None):
    """
    Args:
      cache_location: Where to cache downloaded images. Will be created if it
      doesn't exist.
      batch_size: The size of each batch to load.
      preload_batches: The number of batches that will be preloaded. Increasing
      this number uses more RAM, but can greatly increase performance.
      test_percentage: The percentage of the total images that will be used for
      testing.
      load_datasets_from: The common part of the path to the files that we want
      to load the training and testing datasets from. """
    self._cache = cache.DiskCache(cache_location, 50000000000)
    self._batch_size = batch_size

    self._load_datasets_from = load_datasets_from

    # Initialize datasets.
    self._init_datasets()

  def __del__(self):
    # Save an updated version of our datasets when we exit.
    logger.info("Saving datasets...")
    self.save_datasets()

  def _init_datasets(self):
    """ Initializes the training and testing datasets. """
    if not self._load_datasets_from:
      raise ValueError("load_datasets_from parameter must be a valid path.")

    # Initialize empty datasets.
    self._train_set = dataset.TrainingDataset(set(), self._cache,
                                              self._batch_size,
                                              preload_batches=preload_batches)
    self._test_set = dataset.TestingDataset(set(), self._cache,
                                            self._batch_size,
                                            preload_batches=preload_batches)

    # Use the saved datasets instead of making new ones.
    self.load_datasets()

  def get_random_train_batch(self):
    """ Gets a random training batch.
    Returns:
      The array of loaded images, and the list of labels. """
    images, _ = self._train_set.get_random_batch()

    return images

  def get_random_test_batch(self):
    """ Gets a random testing batch.
    Returns:
      The array of loaded images, and the list of labels. """
    images, _ = self._test_set.get_random_batch()

    return images

  def save_datasets(self):
    """ Saves the datasets to the disk. """
    file_prefix = self._load_datasets_from

    self._train_set.save_images(file_prefix + "_training.pkl")
    self._test_set.save_images(file_prefix + "_testing.pkl")

  def load_datasets(self):
    """ Loads the datasets from the disk. """
    file_prefix = self._load_datasets_from

    self._train_set.load_images(file_prefix + "_training.pkl")
    self._test_set.load_images(file_prefix + "_testing.pkl")
