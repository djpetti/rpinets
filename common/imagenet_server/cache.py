""" Handles caching downloaded data. """


import json
import logging
import operator
import os
import time

import cv2

import numpy as np

import images


logger = logging.getLogger(__name__)


class DiskCache(object):
  """ Caches data to the HDD. """

  def __init__(self, location, max_size):
    """
    Args:
      location: Folder to store the cache in. Will be created if it doesn't
      exist.
      max_size: The maximum size, in bytes, of the cache. """
    # Total size of the cache.
    self.__total_cache_size = 0
    self.__max_size = max_size

    # This is a dict of when we've accessed each file in our cache.
    self.__file_accesses = {}

    self.__location = location
    if not os.path.exists(self.__location):
      os.mkdir(self.__location)

    # Maps synset names to the names of images in them.
    self.__synsets = {}

    self.__load_existing_cache()

  def __image_path(self, synset, image):
    """
    Args:
      synset: The name of the synset the image is in.
      image: The number of the image.
    Returns:
      The path to an image based on its synset and number. """
    image_name = "image%s.jpg" % (image)
    return os.path.join(self.__location, synset, image_name)

  def __load_existing_cache(self):
    """ Accounts for any data that already exists in the cache. """
    for directory in os.listdir(self.__location):
      full_path = os.path.join(self.__location, directory)
      if not os.path.isdir(full_path):
        continue

      # Populate synsets dict.
      self.__synsets[directory] = set([])
      for item in os.listdir(full_path):
        if item.endswith(".jpg"):
          # This is an image.
          self.__synsets[directory].add(item[5:][:-4])
          # Get the size.
          image_path = os.path.join(full_path, item)
          self.__account_for_size(image_path)
          # Get access time.
          self.__file_accesses[image_path] = os.stat(image_path).st_atime

    logger.info("Total cache size: %d", self.__total_cache_size)

  def __account_for_size(self, path):
    """ Adds the size of a file to the total cache size.
    Args:
      path: The path to the file. """
    file_size = os.stat(path).st_size
    self.__total_cache_size += file_size

  def __maintain_size(self):
    """ Makes sure we stay within the bounds of the cache size limit. If we're
    over, it deletes the oldest files until we're at a better size. """
    if self.__total_cache_size <= self.__max_size:
      # We don't have to do anything.
      return

    # Sort files in cache by access time.
    earliest = sorted(self.__file_accesses.items(), key=operator.itemgetter(1))
    # Remove the earliest ones.
    remove_i = 0
    while self.__total_cache_size > self.__max_size:
      path, atime = earliest[remove_i]
      logger.info("Removing %s, last accessed at %f" % (path, atime))
      self.__remove_file(path)
      remove_i += 1

  def __remove_file(self, path):
    """ Removes a file from the cache.
    Args:
      path: The path to the file. """
    # Decrease cache size.
    file_size = os.stat(path).st_size
    self.__total_cache_size -= file_size

    # Remove it from various data structures.
    self.__file_accesses.pop(path)
    path_elements = os.path.normpath(path).split("/")
    synset = path_elements[-2]
    image_number = path_elements[-1][5:][:-4]
    self.__synsets[synset].remove(image_number)

  def add_synset(self, name, words):
    """ Adds a new synset to the cache.
    Args:
      name: The name of the synset.
      words: List of words in the synset. """
    # Make a directory to house the synset.
    location = os.path.join(self.__location, name)
    os.mkdir(location)

    # Store the words for the synset.
    word_file_path = os.path.join(location, "%s.json" % (name))
    word_file = open(word_file_path, "w")
    json.dump(words, word_file)
    word_file.close()

    self.__synsets[name] = set([])

  def add(self, image, name, synset):
    """ Adds a new image to the cache. If the synset is not known, it
    automatically adds that too.
    Args:
      image: The image data to add.
      name: The name of the image.
      synset: The name of the synset to add it to. """
    if synset not in self.__synsets:
      logger.debug("Adding new synset to cache: %s", synset)
      # Get the words for this synset.
      words = images.download_words(synset)
      self.add_synset(synset, words)

    image_path = self.__image_path(synset, name)

    # Write the image data.
    logger.info("Saving new image to cache: %s", image_path)
    cv2.imwrite(image_path, image)

    self.__account_for_size(image_path)
    logger.info("New cache size: %d", self.__total_cache_size)

    self.__synsets[synset].add(name)
    self.__file_accesses[image_path] = time.time()

    # Make sure we stay within the size constraint.
    self.__maintain_size()

  def get(self, synset, image):
    """ Gets an image from the cache.
    Args:
      synset: The name of the synset it belongs to.
      image: The image number in the synset.
    Returns: The image data, or None if the image (or synset) doesn't exist in
             the cache. """
    if synset not in self.__synsets:
      return None
    if image not in self.__synsets[synset]:
      return None

    image_path = self.__image_path(synset, image)

    # Read the image data.
    image = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_UNCHANGED)
    self.__file_accesses[image_path] = time.time()
    return image

class MemoryBuffer(object):
  """ Set of images stored contiguously in memory. This is designed so that it
  can be used as a staging area before a batch is transferred into GPU memory.
  """

  def __init__(self, image_size, batch_size):
    """
    Args:
      image_size: Size of one side of a square image.
      batch_size: How many images are in a batch.
    """
    self.__image_size = image_size
    self.__batch_size = batch_size
    # This will be are underlying storage for the cache.
    self.__storage = np.empty((image_size, batch_size * image_size),
                              dtype="uint8")

    self.__fill_index = 0
    # Maps image names to indices in the underlying array.
    self.__image_indices = {}

  def add(self, image, name):
    """ Adds a new image to the buffer.
    Args:
      image: The image data to add.
      name: The name of the image. """
    logger.debug("Adding %s to buffer at %d." % (name, self.__fill_index))

    next_fill_index = self.__fill_index + self.__image_size
    self.__storage[0:self.__image_size,
                   self.__fill_index:next_fill_index] = image

    self.__image_indices[name] = self.__fill_index
    self.__fill_index = next_fill_index

  def get(self, name):
    """ Gets an image that was added to the buffer.
    Args:
      name: The name of the image.
    Returns:
      The image data. """
    index = self.__image_indices[name]

    return self.__storage[0:self.__image_size, index:index + self.__image_size]

  def get_storage(self):
    """ Returns the entire buffer, so that it can be bulk-loaded. """
    return self.__storage

  def clear(self):
    """ Deletes everything in the cache. """
    self.__fill_index = 0
    self.__image_indices = {}
