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


class Cache(object):
  """ Defines an interface for caches. """

  def add(self, image, name, synset):
    """ Add a new image to the cache.
    Args:
      image: The image to add.
      name: A unique name for the image.
      synset: The synset of the image. """
    raise NotImplementedError("add() must be implemented by subclass.")

  def get(self, synset, name):
    """ Gets an image from the cache.
    Args:
      synset: The synset the image belongs to.
      name: The name of the image. """
    raise NotImplementedError("get() must be implemented by subclass.")


class DiskCache(Cache):
  """ Caches data to the HDD. """

  def __init__(self, location, max_size, download_words=False):
    """
    Args:
      location: Folder to store the cache in. Will be created if it doesn't
      exist.
      max_size: The maximum size, in bytes, of the cache.
      download_words: Whether to download the words for each synset as well as
      just the numbers. """
    super(DiskCache, self).__init__()

    self.__download_words = download_words

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
      found_words = False
      for item in os.listdir(full_path):
        if item.endswith(".jpg"):
          # This is an image.
          self.__synsets[directory].add(item[5:][:-4])
          # Get the size.
          image_path = os.path.join(full_path, item)
          self.__account_for_size(image_path)
          # Get access time.
          self.__file_accesses[image_path] = os.stat(image_path).st_atime

        if item.endswith(".json"):
          # List of words for the synset.
          found_words = True

      # If we need to download synset words, and we haven't already, do that
      # now.
      if (self.__download_words and not found_words):
        self.__download_words_for_synset(directory)

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

  def __add_synset(self, name):
    """ Adds a new synset to the cache.
    Args:
      name: The name of the synset. """
    # Make a directory to house the synset.
    location = os.path.join(self.__location, name)
    os.mkdir(location)

    # Download words if we need to.
    if self.__download_words:
      self.__download_words_for_synset(name)

    self.__synsets[name] = set([])

  def __download_words_for_synset(self, synset):
    """ Downloads the words that correspond to a particular synset.
    Args:
      synset: The name of the synset to download words for. """
    # Get the words for this synset.
    words = images.download_words(synset)

    # Store the words for the synset.
    location = os.path.join(self.__location, synset)
    word_file_path = os.path.join(location, "%s.json" % (synset))
    word_file = open(word_file_path, "w")
    json.dump(words, word_file)
    word_file.close()

  def add(self, image, name, synset):
    """ Adds a new image to the cache. If the synset is not known, it
    automatically adds that too.
    Args:
      image: The image data to add.
      name: The name of the image.
      synset: The name of the synset to add it to. """
    if synset not in self.__synsets:
      logger.debug("Adding new synset to cache: %s", synset)
      self.__add_synset(synset)

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

  def get(self, synset, name):
    """ Gets an image from the cache.
    Args:
      synset: The name of the synset it belongs to.
      name: The image name in the synset.
    Returns: The image data, or None if the image (or synset) doesn't exist in
             the cache. """
    if synset not in self.__synsets:
      return None
    if name not in self.__synsets[synset]:
      return None

    image_path = self.__image_path(synset, name)

    # Read the image data.
    image = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_UNCHANGED)
    self.__file_accesses[image_path] = time.time()

    return image

class MemoryBuffer(Cache):
  """ Set of images stored contiguously in memory. This is designed so that it
  can be used as a staging area before a batch is transferred into GPU memory.
  """

  def __init__(self, image_size, batch_size, num_batches, channels=1,
               num_patches=1):
    """
    Args:
      image_size: Size of one side of a square image.
      batch_size: The size of each batch of images.
      num_batches: The number of batches we should be able to hold.
      channels: Number of channels the images have.
      num_patches: The number of patches that are expected for each image.
    """
    super(MemoryBuffer, self).__init__()

    self.__image_size = image_size
    self.__batch_size = batch_size
    self.__num_batches = num_batches
    self.__num_patches = num_patches

    self.__num_images = self.__batch_size * self.__num_batches * \
                        self.__num_patches
    logger.debug("Creating mem buffer with total capacity %d." % \
                 (self.__num_images))

    # This will be are underlying storage for the cache.
    self.__channels = channels
    shape = (self.__num_images, self.__channels, image_size, image_size)
    self.__storage = np.empty(shape, dtype="uint8")

    self.__fill_index = 0
    # Fill index for the label array.
    self.__label_fill_index = 0
    # Maps image names to indices in the underlying array.
    self.__image_indices = {}
    # Keeps a list of image synsets in the order that they were added.
    self.__labels = [None] * self.__batch_size * self.__num_batches

    # Keeps track of the start of the last batch that we got.
    self.__batch_index = 0
    # Batch index for the label array.
    self.__label_batch_index = 0

    # Total number of images in the buffer.
    self.__data_in_buffer = 0

  def __increment_fill_index(self):
    """ Increments the fill index, handling boundary conditions appropriately.
    """
    self.__fill_index += 1
    self.__label_fill_index += 1

    if self.__fill_index % self.__batch_size == 0:
      # We added a complete batch, so we have to skip a bunch of indices to
      # account for multiple patches before we start adding the next batch.
      self.__fill_index += (self.__num_patches - 1) * self.__batch_size

    # It should wrap back to zero when we hit the end.
    self.__fill_index %= self.__num_images
    self.__label_fill_index %= self.__batch_size * self.__num_batches

    self.__data_in_buffer += self.__num_patches
    if self.__data_in_buffer > self.get_max_patches():
      raise ValueError("Cannot add to full buffer.")

  def __increment_batch_index(self):
    """ Increments the batch index, handling boundary conditions appropriately. """
    self.__batch_index += self.__batch_size * self.__num_patches
    self.__label_batch_index += self.__batch_size

    # It should wrap back to zero at the end.
    self.__batch_index %= self.__num_images
    self.__label_batch_index %= self.__batch_size * self.__num_batches

    self.__data_in_buffer -= self.__batch_size * self.__num_patches
    if self.__data_in_buffer < 0:
      raise ValueError("Not enough data in buffer for a complete patch.""")

  def add(self, image, name, synset):
    """ Adds a new image to the buffer.
    Args:
      image: The image data to add.
      name: The name of the image.
      synset: The synset of the image. """
    self.__storage[self.__fill_index] = np.transpose(image, (2, 0, 1))

    unique_identifier = "%s_%s" % (synset, name)
    self.__image_indices[unique_identifier] = self.__fill_index

    self.__labels[self.__label_fill_index] = synset

    self.__increment_fill_index()

  def add_patches(self, patches, name, synset):
    """ Similar to add, except that it adds multiple patches for the same image.
    Args:
      patches: The list of patches to store.
      name: The name of the image.
      synset: The synset of the image. """
    if len(patches) != self.__num_patches:
      raise ValueError("Expected %d patches, got %d." % (self.__num_patches,
                                                         len(patches)))

    patch_locations = []
    for i, patch in enumerate(patches):
      location = self.__fill_index + i * self.__batch_size
      self.__storage[location] = np.transpose(patch, (2, 0, 1))
      patch_locations.append(location)


    unique_identifier = "%s_%s" % (synset, name)
    self.__image_indices[unique_identifier] = patch_locations

    self.__labels[self.__label_fill_index] = synset

    self.__increment_fill_index()

  def get(self, synset, name):
    """ Gets an image that was added to the buffer. If there are multiple
    patches of this image, it returns a list of all the patches.
    Args:
      name: The name of the image.
    Returns:
      The image data. """
    def get_image(index):
      """ Get the image at the specified index.
      Args:
        index: The index of the image.
      Returns:
        The image data. """
      return self.__storage[0:self.__image_size,
                            index:index + self.__image_size,
                            0:self.__channels]


    unique_identifier = "%s_%s" % (synset, name)
    index = self.__image_indices[unique_identifier]

    if type(index) is list:
      # There are multiple patches.
      patches = []
      for i in index:
        patches.append(get_image(i))
      return patches

    return get_image(index)

  def get_storage(self):
    """ Returns the entire buffer, so that it can be bulk-loaded, as well as the
    labels for every item in the buffer. """
    return (self.__storage, self.__labels)

  def get_batch(self):
    """ Gets a portion of the storage of size batch_size. After this is called,
    it advances batch_index, so that this portion of memory can be overwritten
    again.
    Returns:
      The batch data, and label data. """
    end_index = self.__batch_index + self.__batch_size * self.__num_patches
    logger.debug("Getting batch %d:%d." % (self.__batch_index, end_index))
    batch = self.__storage[self.__batch_index:end_index]

    # We only have one label, even if we have multiple patches.
    end_labels = self.__label_batch_index + self.__batch_size
    logger.debug("Getting batch labels: %d:%d." % (self.__label_batch_index,
                                                   end_labels))
    labels = self.__labels[self.__label_batch_index:end_labels]

    self.__increment_batch_index()

    return batch, labels

  def clear(self):
    """ Deletes everything in the cache. """
    self.__fill_index = 0
    self.__batch_index = 0
    self.__data_in_buffer = 0
    self.__image_indices = {}

  def get_max_patches(self):
    """
    Returns:
      The maximum number of images that can be stored in this buffer, counting
      all patches individually. """
    return self.__num_images

  def get_max_images(self):
    """
    Returns:
      The maximum number of images that can be stored in this buffer, all the
      patches of a single image as one image. """
    return self.__batch_size * self.__num_batches

  def space_used(self):
    """
    Returns:
      The total number of images in this buffer. """
    return self.__data_in_buffer

  def space_remaining(self):
    """
    Returns:
      The total number of images that can still be added to this buffer before
      it's full. """
    return self.get_max_patches() - self.__data_in_buffer
