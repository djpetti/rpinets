import logging

import cache
import dataset
import downloader


logger = logging.getLogger(__name__)


class ImageGetter(object):
  """ Gets random sets of images for use in training and testing. """

  def __init__(self, cache_location, batch_size, image_shape,
               preload_batches=1, test_percentage=0.1,
               load_datasets_from=None, patch_shape=None, patch_flip=True,
               link_with=[], pca_stddev=25, jitter_stddev=0):
    """
    Args:
      cache_location: Where to cache downloaded images. Will be created if it
      doesn't exist.
      batch_size: The size of each batch to load. It can optionally be a
                  two-element tuple, which specifies sizes for training and
                  testing batches, respectively.
      image_shape: A three-element tuple containing the x and y size of the raw
                   images that will be handled, and the number of channels.
      preload_batches: The number of batches that will be preloaded. Increasing
                        this number uses more RAM, but can greatly increase
                        performance.
      test_percentage: The percentage of the total images that will be used for
                       testing.
      load_datasets_from: The common part of the path to the files that we want
                          to load the training and testing datasets from.
      patch_shape: The shape of the patches to extract from each image. If this
                   is None, no patches will be extracted, and the raw images
                   will be used directly. Furthermore, if this is specified, the
                   batches from the testing dataset will contain copies of every
                   patch. The exception to this is when using linked datasets,
                   in which case only the main dataset will be patched, and the
                   testing one will contain no extra copies. (The auxiliary
                   datasets will be reshaped to maintain the same size.)
      patch_flip: Whether to include horizontally flipped patches.
      link_with: Specifies a list of cache directories to link with the current
                 dataset.
      pca_stddev: The standard deviation for PCA.
      jitter_stddev: The standard deviation for jitter."""
    if len(image_shape) != 3:
      raise ValueError( \
          "Expected image shape of form (x size, y size, channels).")
    if patch_shape and len(patch_shape) != 2:
      raise ValueError( \
          "Expected patch shape of form (x size, y size).")

    self._cache = cache.DiskCache(cache_location, 50000000000)
    if hasattr(batch_size, "__getitem__"):
      # The user passed in a tuple.
      self._train_batch_size, self._test_batch_size = batch_size
    else:
      # One global batch size.
      self._train_batch_size = self._test_batch_size = batch_size

    self._image_shape = image_shape
    self._patch_shape = patch_shape
    self._patch_flip = patch_flip

    self._preload_batches = preload_batches
    self._test_percentage = test_percentage
    self._load_datasets_from = load_datasets_from

    self._pca_stddev = pca_stddev
    self._jitter_stddev = jitter_stddev

    self.__loaded_datasets = False
    self.__cleaned_up = False

    self.__link_with = link_with

    # Initialize datasets.
    self._init_datasets()

  def __del__(self):
    self.cleanup()

  def cleanup(self):
    """ Cleans up the image getter. This has to get run at some point, so it's
    good practice to call it, even though it's called by the destructor too. """
    if self.__cleaned_up:
      # We already did this. We don't have to do it again.
      return

    # Save an updated version of our datasets when we exit.
    if self.__loaded_datasets:
      logger.info("Saving datasets...")
      self.save_datasets()

    # Stop downloader processes.
    downloader.cleanup()

    self.__cleaned_up = True

  def _make_new_datasets(self, train_data, test_data):
    """ Make training and testing datasets.
    Args:
      train_data: Data for training set.
      test_data: Data for testing set. """
    # Load all the caches we requested.
    all_caches = [self._cache]
    for cache_name in self.__link_with:
      all_caches.append(cache.DiskCache(cache_name))

    # Training dataset.
    if self.__link_with:
      # Build linked dataset.
      self._train_set = dataset.LinkedDataset(train_data, all_caches,
                                              self._train_batch_size,
                                              self._image_shape,
                                              preload_batches=self._preload_batches,
                                              patch_shape=self._patch_shape,
                                              patch_flip=self._patch_flip,
                                              pca_stddev=self._pca_stddev,
                                              jitter_stddev=self._jitter_stddev)
    else:
      # No special features.
      self._train_set = dataset.Dataset(train_data, self._cache,
                                        self._train_batch_size,
                                        self._image_shape,
                                        preload_batches=self._preload_batches,
                                        patch_shape=self._patch_shape,
                                        patch_flip=self._patch_flip,
                                        pca_stddev=self._pca_stddev,
                                        jitter_stddev=self._jitter_stddev)

    # Testing dataset.
    if self.__link_with:
      # Build linked dataset.
      self._test_set = dataset.LinkedDataset(test_data, all_caches,
                                             self._test_batch_size,
                                             self._image_shape,
                                             preload_batches= \
                                                self._preload_batches,
                                             patch_shape=self._patch_shape,
                                             patch_flip=self._patch_flip,
                                             pca_stddev=self._pca_stddev,
                                             jitter_stddev=self._jitter_stddev)
    elif self._patch_shape:
      # Use all the patches in the test set.
      self._test_set = dataset.PatchedDataset(test_data, self._cache,
                                              self._test_batch_size,
                                              self._image_shape,
                                              preload_batches= \
                                                  self._preload_batches,
                                              patch_shape=self._patch_shape,
                                              patch_flip=self._patch_flip,
                                              pca_stddev=self._pca_stddev,
                                              jitter_stddev=self._jitter_stddev)
    else:
      # No special features.
      self._test_set = dataset.Dataset(test_data, self._cache,
                                       self._test_batch_size, self._image_shape,
                                       preload_batches=self._preload_batches,
                                       patch_shape=self._patch_shape,
                                       patch_flip=self._patch_flip,
                                       pca_stddev=self._pca_stddev,
                                       jitter_stddev=self._jitter_stddev)

  def _init_datasets(self):
    """ Initializes the training and testing datasets. """
    if not self._load_datasets_from:
      raise ValueError("load_datasets_from parameter must be a valid path.")

    # Initialize empty datasets.
    self._make_new_datasets(set(), set())
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

  def get_sequential_train_batch(self):
    """ Gets a sequential training batch.
    Returns:
      The array of loaded images, and the list of labels. """
    return self._train_set.get_sequential_batch()

  def get_sequential_test_batch(self):
    """ Gets a sequential testing batch.
    Returns:
      The array of loaded images, and the list of labels. """
    return self._test_set.get_sequential_batch()

  def get_specific_train_batch(self, images):
    """ Gets a specific training batch.
    Args:
      images: The specific images to get.
    Returns:
      The array of loaded images, and the list of labels. """
    return self._train_set.get_specific_batch(images)

  def get_specific_test_batch(self, images):
    """ Gets a specific testing batch.
    Args:
      images: The specific images to get.
    Returns:
      The array of loaded images, and the list of labels. """
    return self._test_set.get_specific_batch(images)

  def get_train_set_size(self):
    """
    Returns:
      The number of images in the training dataset. """
    return len(self._train_set.get_images())

  def get_test_set_size(self):
    """
    Returns:
      The number of images in the testing dataset. """
    return len(self._test_set.get_images())

  def get_train_image_names(self):
    """
    Returns:
      The names of the images in the training dataset. """
    return self._train_set.get_images()

  def get_test_image_names(self):
    """ Returns:
      The names of the images in the testing dataset. """
    return self._test_set.get_images()

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

    self.__loaded_datasets = True
