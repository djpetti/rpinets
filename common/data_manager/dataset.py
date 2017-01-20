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
               patch_shape=None, patch_flip=True):
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
                   be used instead.
      patch_flip: Whether to include flipped patches. """
    self._image_shape = image_shape

    # Split image data into a set with just the IDs, and a map of the IDs to the
    # URLs. For the IDs, we will use the special RandomSet type so we can
    # quickly choose random images.
    image_data = []
    self.__image_urls = {}
    for img_id, url in images:
      image_data.append(img_id)
      self.__image_urls[img_id] = url
    self.__images = RandomSet(image_data)

    self._cache = disk_cache

    self._batch_size = batch_size
    self._patch_shape = patch_shape
    self._patch_flip = patch_flip

    # The fraction of the last batch that it was able to find in the cache.
    self.__cache_hit_rate = 0.0

    # The image ID of the first image in the cache.
    self.__first_cached_image = None
    # The image to start at when loading the next sequential batch.
    self.__sequential_start_image = None

    logger.info("Have %d total images in database." % (len(self.__images)))

  def _pick_random_image(self):
    """ Picks a random image from our database.
    Returns:
      img_id and url, and key of the image in the images map. """
    # Pick a random image.
    img_id = self.__images.get_random()
    url = self.__image_urls[img_id]

    if img_id in self.__already_picked:
      # This is a duplicate, pick another.
      return self._pick_random_image()
    self.__already_picked.add(img_id)

    return img_id, url

  def get_random_batch(self):
    """ Loads a random batch of images from the whole dataset.
    Returns:
      The array of loaded images, a list of the label each image has, and a list
      of the image IDs for each image, as well as a list of all images that
      failed to download. """
    logger.info("Getting batch.")

    # Keeps track of images that were already used this batch.
    self.__already_picked = set([])

    # Try to download initial images.
    need_images = self._mem_buffer.space_remaining() - \
                  self._download_manager.get_num_downloading()
    logger.debug("Need to start downloading %d additional images." %
                 (need_images))

    total_cache_hits = 0
    cache_attempted_loads = 0
    for _ in range(0, need_images):
      total_cache_hits += self._load_random_images(need_images)
      cache_attempted_loads += 1
      need_images = self._mem_buffer.space_remaining()

      if not need_images:
        # We loaded everything from cache.
        logger.debug("Loaded everything from cache.")
        break

    # Wait for 1 batch worth of the downloads to complete,
    # replacing any ones that fail.
    to_remove = []
    while self._mem_buffer.space_used() < self._batch_size:
      # Update the download manager.
      downloaded = self._download_manager.update()

      # Process failures.
      failures = self._download_manager.get_failures()
      if len(failures):
        logger.debug("Replacing %d failed downloads." % (len(failures)))

      # Remove failed images.
      for synset, name, url in failures:
        # Remove failed image.
        img_id = utils.make_img_id(synset, name)

        logger.info("Removing bad image %s, %s" % (img_id, url))

        # Because we keep running downloader processes in the background through
        # multiple calls to get_random_batch, it's possible that we could try to
        # remove something twice.
        if img_id in self.__images:
          self.__images.remove(img_id)
          self.__image_urls.pop(img_id)
          to_remove.append((img_id, url))
        else:
          logger.debug("Item already removed: %s" % (img_id))

        # Load new images to replace it.
        total_cache_hits += self._load_random_images( \
            self._mem_buffer.space_remaining())
        cache_attempted_loads += 1

      time.sleep(0.2)

    # Update total hit rate.
    if cache_attempted_loads:
      self.__cache_hit_rate = float(total_cache_hits) / cache_attempted_loads
    logger.debug("Cache hits: %d, Hit rate: %f" % (total_cache_hits,
                                                   self.__cache_hit_rate))

    return self._mem_buffer.get_batch(), to_remove

  def get_sequential_batch(self):
    """ Loads images sequentially from the cache instead of picking a random
    batch. This is probably not something you want to do unless you have a very
    large fraction of your dataset in cache to begin with.
    Returns:
      The array of loaded images, and a list of all the image labels. """
    logger.info("Getting sequential batch.")

    if not self.__first_cached_image:
      # Find the first image in the cache.
      self.__first_cached_image = \
          self._cache.get_first_in_cache(use_only=self.__images)
      logger.debug("First image in cache: %s" % (self.__first_cached_image))

      self.__sequential_start_image = self.__first_cached_image

    # Load the next sequential batch.
    loaded_nothing = False
    while self._mem_buffer.space_used() < self._batch_size:
      start_label, start_name = \
          utils.split_img_id(self.__sequential_start_image)

      # Try loading the images.
      need_images = self._mem_buffer.space_remaining()
      loaded, next_start = self._get_cached_images(start_label, start_name,
                                                   need_images,
                                                   force_sequential=True)
      if not next_start:
        # We didn't get any images, which means we reached the end of the cache.
        # Go back to the beginning.
        logger.debug("Reached end of cache.")
        self.__sequential_start_image = self.__first_cached_image
      else:
        # Just start at the next batch.
        logger.debug("New start image: %s" % (next_start))
        self.__sequential_start_image = next_start

      if not loaded:
        if loaded_nothing:
          # This is the second cycle in a row where we've gotten nothing, so
          # something is clearly wrong.
          raise RuntimeError("Not enough images to load from cache?")
        loaded_nothing = True
      else:
        loaded_nothing = False

    return self._mem_buffer.get_batch()

  def get_specific_batch(self, images):
    """ Loads a set of specified images and returns them as a batch.
    Args:
      images: The images to load, in the form of (label, name) tuples.
    Returns:
      The array of loaded images, a list of all the image labels, and a list of
      the images that were not found. """
    logger.info("Getting specific batch.")

    # We can simply bulk-load all the images.
    loaded, not_found = self._cache.bulk_get(images)
    for img_id, image in loaded.iteritems():
      self._buffer_image(image, img_id)

    return self._mem_buffer.get_batch()

  def _load_random_images(self, max_images):
    """ Loads a random image from either the cache or the internet. If loading
    from the cache, it will also try to speed up the process by loading "bonus"
    images that are physically near the target image in the cache, if deemed
    reasonable.
    Args:
      max_images: The maximum number of images to load.
    Returns:
      How many images it loaded from the cache successfully. """
    img_id, url = self._pick_random_image()
    synset, number = utils.split_img_id(img_id)

    if self._cache.is_in_cache(synset, number):
      # It's in the cache, so load it and any additional sequential images.
      loaded, _ = self._get_cached_images(synset, number, max_images)
      return loaded

    # We have to download the image instead.
    if not self._download_manager.download_new(synset, number, url):
      # We're already downloading that image.
      logger.info("Already downloading %s. Picking new one..." % (img_id))
      return self._load_random_images(max_images)

    return 0

  def _get_cached_images(self, start_label, start_name, max_images,
                         force_sequential=False):
    """ Bulk-loads a bunch of image data from the cache, pre-processes them, and
    puts them in the memory buffer.
    Args:
      start_label: The label of the image to start loading from.
      start_name: The name of the image to start loading from.
      max_images: The maximum number of images we can load.
      force_sequential: Ignore heuristics and force sequential loading all the
                        time.
    Returns:
      1 if it loaded something from the cache, 0 if it didn't. It also returns
      where to start loading the next sequential batch, if one can be loaded. """
    # Use a simple heuristic to figure out how many images to try loading
    # sequentially.
    if not force_sequential:
      load_images = self.__cache_hit_rate * max_images
      load_images = int(load_images)
      load_images = max(load_images, 1)
    else:
      load_images = max_images
    logger.debug("Attempting load of %d sequential images from cache..." % \
                 (load_images))

    loaded, next_start = self._cache.get_sequential(start_label, start_name,
                                                    load_images,
                                                    use_only=self.__images)

    for img_id, image in loaded.iteritems():
      self._buffer_image(image, img_id)

    return (len(loaded) != 0, next_start)

  def _buffer_image(self, image, img_id):
    """ Pre-process a loaded image and store it in the memory buffer.
    Args:
      image: The actual image data to store.
      img_id: The ID of the image. """
    # Select a patch.
    if self._patch_shape:
      patches = data_augmentation.extract_patches(image, self._patch_shape,
                                                  flip=self._patch_flip)
      image = patches[random.randint(0, len(patches) - 1)]

    # Add it to the buffer.
    label, name = utils.split_img_id(img_id)
    self._mem_buffer.add(image, name, label)

  def get_images(self):
    """ Gets all the images in the dataset. """
    return self.__images

  def save_images(self, filename):
    """ Saves the set of images that this dataset contains to a file.
    Args:
      filename: The name of the file to write the list of images to. """
    image_file = open(filename, "wb")
    logger.debug("Saving dataset to file: %s" % (filename))
    pickle.dump((self.__images, self.__image_urls), image_file)
    image_file.close()

  def load_images(self, filename):
    """ Loads the set of images that this dataset contains from a file.
    Args:
      filename: The name of the file to read the list of images from. """
    image_file = file(filename, "rb")
    logger.info("Loading dataset from file: %s" % (filename))
    self.__images, self.__image_urls = pickle.load(image_file)

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
      for img_id, _ in labels[label]:
        valid.add(img_id)

    # Remove anything that's not in it.
    removed_image = False
    to_remove = []
    for img_id in self.__images:
      if img_id not in valid:
        to_remove.append(img_id)
        removed_image = True

    for img_id in to_remove:
      self.__images.remove(img_id)

    return removed_image


class Dataset(_DatasetBase):
  """ A standard dataset. """

  def __init__(self, images, disk_cache, batch_size, image_shape,
               preload_batches=1, patch_shape=None, patch_flip=True):
    """ See documentation for superclass method.
    Extra Args:
      preload_batches: How many additional batches to preload. This can greatly
                       increase performace, but uses additional RAM. """
    super(Dataset, self).__init__(images, disk_cache, batch_size, image_shape,
                                  patch_shape=patch_shape,
                                  patch_flip=patch_flip)

    # We need to size the buffer accordingly for what size images are
    # going to be stored.
    buffer_shape = self._image_shape[:2]
    _, _, channels = self._image_shape
    if self._patch_shape:
      buffer_shape = self._patch_shape

    self._mem_buffer = cache.MemoryBuffer(buffer_shape, batch_size,
                                          preload_batches, channels=channels)
    self._download_manager = \
        downloader.DownloadManager(self._cache,
                                   self._mem_buffer, image_shape,
                                   patch_shape=self._patch_shape)


class PatchedDataset(_DatasetBase):
  """ Dataset that extracts every patch for each image, and stores
  one version of each batch for every patch type. """

  def __init__(self, images, disk_cache, batch_size, image_shape,
               preload_batches=1, patch_shape=None, patch_flip=True):
    """ See documentation for _DatasetBase __init__ function.
    NOTE: batch_size here represents the base batch size. When you request a
    batch, it will actually return 10 times this many images, since it will use
    all the patches.
    Extra Args:
      preload_batches: How many additional batches to preload. This can greatly
                       increase performace, but uses additional RAM. """
    if not patch_shape:
      raise ValueError("Keyword arg patch_shape is required for PatchedDataset.")

    super(PatchedDataset, self).__init__(images, disk_cache, batch_size,
                                         image_shape, patch_shape=patch_shape,
                                         patch_flip=patch_flip)

    # We need extra buffer space to store the extra patches.
    num_patches = 10
    if not patch_flip:
      # We're not storing flipped patches, which halves our storage
      # requirements.
      num_patches /= 2
    _, _, channels = self._image_shape
    self._mem_buffer = cache.MemoryBuffer(self._patch_shape, batch_size,
                                          preload_batches, channels=channels,
                                          num_patches=num_patches)
    self._download_manager = \
        downloader.DownloadManager(self._cache,
                                   self._mem_buffer, image_shape,
                                   all_patches=True,
                                   patch_shape=self._patch_shape)

  def _buffer_image(self, image, img_id):
    """ Pre-process a loaded image and store it in the memory buffer.
    Args:
      image: The actual image data to store.
      img_id: The ID of the image. """
    # Add all the patches.
    patches = data_augmentation.extract_patches(image, self._patch_shape,
                                                flip=self._patch_flip)

    label, name = utils.split_img_id(img_id)
    self._mem_buffer.add_patches(patches, name, label)
