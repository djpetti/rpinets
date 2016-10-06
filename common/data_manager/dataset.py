import logging
import cPickle as pickle
import os
import random
import time

from randomset import RandomSet
import cache
import data_augmentation
import downloader
import utils


logger = logging.getLogger(__name__)


class _DatasetBase(object):
  """ Represents a single dataset, with a fixed group of images that it draws
  randomly from. """

  def __init__(self, images, disk_cache, batch_size, image_shape,
               patch_shape=None):
    """
    Args:
      images: The list of images that makes up this dataset. The list should
      contain tuples with each image's img_id and URL. If the URL is None, then
      that image should already be present in the cache, otherwise it will cause
      an error.
      disk_cache: The disk cache where we can store downloaded images.
      batch_size: The size of each batch from this dataset.
      image_shape: A three-element tuple containing the x and y shape of the
                   raw images, and the number of channels.
      patch_shape: The shape of the patches to extract from images. If it is
                   None, no patches will be extracted, and the base image will
                   be used instead."""
    self._image_shape = image_shape

    self.__images = RandomSet(images)
    self._cache = disk_cache

    self._batch_size = batch_size
    self._patch_shape = patch_shape

    # The number of images that we've currently requested be downloaded.
    self.__num_downloading = 0

    logger.info("Have %d total images in database." % (len(self.__images)))

  def _pick_random_image(self):
    """ Picks a random image from our database.
    Returns:
      img_id and url, and key of the image in the images map. """
    # Pick a random image.
    img_id, url = self.__images.get_random()

    if img_id in self.__already_picked:
      # This is a duplicate, pick another.
      return self._pick_random_image()
    self.__already_picked.add(img_id)

    return img_id, url

  def get_random_batch(self):
    """ Loads a random batch of images from the whole dataset.
    Returns:
      The array of loaded images, and a list of the labels each image belongs
      to, as well as a list of all images that failed to download. """
    logger.info("Getting batch.")

    # Keeps track of images that were already used this batch.
    self.__already_picked = set([])

    # Try to download initial images.
    need_images = self._mem_buffer.get_max_images() - self.__num_downloading
    logger.debug("Need to start downloading %d additional images." %
                 (need_images))
    loaded_from_cache = 0
    for _ in range(0, need_images):
      loaded_from_cache += self._load_random_image()
    self.__num_downloading += need_images

    # Wait for 1 batch worth of the downloads to complete,
    # replacing any ones that fail.
    to_remove = []
    successfully_downloaded = loaded_from_cache
    while successfully_downloaded < self._batch_size:
      failures = self._download_manager.get_failures()
      if len(failures):
        logger.debug("Replacing %d failed downloads." % (len(failures)))

      # Remove failed images.
      loaded_from_cache = 0
      for label, name, url in failures:
        # Remove failed image.
        img_id = utils.make_img_id(label, name)

        logger.info("Removing bad image %s, %s" % (img_id, url))

        # Because we keep running downloader processes in the background through
        # multiple calls to get_random_batch, it's possible that we could try to
        # remove something twice.
        if (img_id, url) in self.__images:
          self.__images.remove((img_id, url))
          to_remove.append((img_id, url))
        else:
          logger.debug("Item already removed: %s" % (img_id))

        # Load a new image to replace it.
        loaded_from_cache += self._load_random_image()

      downloaded = self._download_manager.update()
      successfully_downloaded += downloaded + loaded_from_cache

      time.sleep(0.2)

    logger.debug("Loaded %d images from cache." % (loaded_from_cache))
    logger.debug("Finished downloading.")
    self.__num_downloading -= self._batch_size
    return self._mem_buffer.get_batch(), to_remove

  def _load_random_image(self):
    """ Loads a random image from either the cache or the internet. If loading
    from the internet, it adds it to the download manager, otherwise, it adds
    them to the memory buffer.
    Returns:
      How many images it loaded from the cache successfully. """
    img_id, url = self._pick_random_image()
    label, number = utils.split_img_id(img_id)

    image = self._get_cached_image(label, number)
    if image is None:
      if not url:
        # Now we have a problem, because we can't get the image anywhere.
        logger.error("Expected image %s to be in cache, but it is not there." \
                     % (img_id))
        raise ValueError("Expected %s to be in cache." % (img_id))

      # We have to download the image instead.
      if not self._download_manager.download_new(label, number, url):
        # We're already downloading that image.
        logger.info("Already downloading %s. Picking new one..." % (img_id))
        return self._load_random_image()
      return 0

    # Cache hit.
    self._mem_buffer.add(image, number, label)
    return 1

  def _get_cached_image(self, label, image_number):
    """ Checks if an image is in the cache, and returns it.
    Args:
      label: The label of the image.
      image_number: The image number in the label.
    Returns:
      The actual image data, or None if the image is not in the cache. """
    # First check in the cache for the image.
    cached_image = self._cache.get(label, image_number)
    if cached_image is not None:
      # We had a cached copy, so we're done.
      if self._patch_shape:
        # Select a patch.
        patches = data_augmentation.extract_patches(cached_image,
                                                    self._patch_shape)
        cached_image = patches[random.randint(0, len(patches) - 1)]

      return cached_image

    return None

  def get_images(self):
    """ Gets all the images in the dataset. """
    return self.__images

  def save_images(self, filename):
    """ Saves the set of images that this dataset contains to a file.
    Args:
      filename: The name of the file to write the list of images to. """
    image_file = open(filename, "wb")
    logger.debug("Saving dataset to file: %s" % (filename))
    pickle.dump(self.__images, image_file)
    image_file.close()

  def load_images(self, filename):
    """ Loads the set of images that this dataset contains from a file.
    Args:
      filename: The name of the file to read the list of images from. """
    if not os.path.exists(filename):
      raise ValueError("Dataset file does not exist: %s" % (filename))

    image_file = file(filename, "rb")
    logger.info("Loading dataset from file: %s" % (filename))
    self.__images = pickle.load(image_file)

  def prune_images(self, labels):
    """ Remove any images from the dataset that are not specified in the input.
    Args:
      labels: The label data structure containing all the acceptable images.
      If something is not in here, it will be removed.
    Returns:
      True if images were removed, False if none were. """
    logger.info("Pruning dataset...")

    # Put everything into a valid set.
    valid = set()
    for label in labels.keys():
      for img_id, url in labels[label]:
        valid.add((img_id, url))

    # Remove anything that's not in it.
    removed_image = False
    for img_id, url in self.__images:
      if (img_id, url) not in valid:
        self.__images.remove((img_id, url))
        removed_image = True

    return removed_image


class Dataset(_DatasetBase):
  """ A standard dataset. """

  def __init__(self, images, disk_cache, batch_size, image_shape,
               preload_batches=1, patch_shape=None):
    """ See documentation for superclass method.
    Extra Args:
      preload_batches: How many additional batches to preload. This can greatly
                       increase performace, but uses additional RAM.
      patch_shape: A two-element tuple containing the size of each patch. If it
                   is None, no patches will be extracted, and the image will be
                   saved directly. """
    super(Dataset, self).__init__(images, disk_cache, batch_size, image_shape,
                                  patch_shape=patch_shape)

    # We need to size the buffer accordingly for what images sized images are
    # going to be stored.
    buffer_shape = self._image_shape[:2]
    _, _, channels = self._image_shape
    if self._patch_shape:
      buffer_shape = self._patch_shape

    self._mem_buffer = cache.MemoryBuffer(buffer_shape, batch_size,
                                          preload_batches, channels=channels)
    self._download_manager = \
        downloader.DownloadManager(self._cache,
                                   self._mem_buffer,
                                   patch_shape=self._patch_shape)


class PatchedDataset(_DatasetBase):
  """ Dataset that extracts every patch for each image, and stores
  10 versions of each batch, one for each patch type. """

  def __init__(self, images, disk_cache, batch_size, image_shape,
               preload_batches=1, patch_shape=None):
    """ See documentation for _DatasetBase __init__ function.
    NOTE: batch_size here represents the base batch size. When you request a
    batch, it will actually return 10 times this many images, since it will use
    all the patches.
    Extra Args:
      patch_shape: A two-element tuple containing the size of each patch.
      preload_batches: How many additional batches to preload. This can greatly
                       increase performace, but uses additional RAM. """
    if not patch_shape:
      raise ValueError("Keyword arg patch_shape is required for PatchedDataset.")

    super(PatchedDataset, self).__init__(images, disk_cache, batch_size,
                                         image_shape, patch_shape=patch_shape)

    # We need 10x the buffer space to store the extra patches.
    _, _, channels = self._image_shape
    self._mem_buffer = cache.MemoryBuffer(self._patch_shape, batch_size,
                                          preload_batches, channels=channels,
                                          num_patches=10)
    self._download_manager = \
        downloader.DownloadManager(self._cache,
                                   self._mem_buffer,
                                   all_patches=True,
                                   patch_shape=self._patch_shape)

  def _load_random_image(self):
    """ See superclass documentation. This override is necessary to deal with
    multiple patches. """
    img_id, url = self._pick_random_image()
    label, number = utils.split_img_id(img_id)

    patches = self._get_cached_image(label, number)
    if patches is None:
      # We have to download the image instead.
      self._download_manager.download_new(label, number, url)
      return 0

    # Cache hit.
    self._mem_buffer.add_patches(patches, number, label)
    return 1

  def _get_cached_image(self, label, image_number):
    """ See superclass documentation. This override is necessary to deal with
    multiple patches. """
    # First check in the cache for the image.
    cached_image = self._cache.get(label, image_number)
    if cached_image is not None:
      # Extract patches.
      return data_augmentation.extract_patches(cached_image)

    return None
