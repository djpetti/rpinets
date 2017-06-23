""" Handles caching downloaded data. """

import collections
import json
import logging
import operator
import os
import cPickle as pickle
import sys
import time

import cv2

import numpy as np

import images
import utils


logger = logging.getLogger(__name__)


class Cache(object):
  """ Defines an interface for caches. """

  def add(self, image, name, label):
    """ Add a new image to the cache.
    Args:
      image: The image to add.
      name: A unique name for the image.
      label: The label of the image. """
    raise NotImplementedError("add() must be implemented by subclass.")

  def get(self, label, name):
    """ Gets an image from the cache.
    Args:
      label: The label the image belongs to.
      name: The name of the image. """
    raise NotImplementedError("get() must be implemented by subclass.")


class DiskCache(Cache):
  """ Caches data to the HDD. """

  def __init__(self, location, max_size=None, extension=".jpg"):
    """
    Args:
      location: Folder to store the cache in. Will be created if it doesn't
      exist.
      max_size: The maximum size, in bytes, of the cache. If not specified,
                there will be no maximum size.
      extension: The extension that we use for saved data, in this case, it
      mainly defines the compression. It defaults to JPEG. """
    super(DiskCache, self).__init__()

    self.__extension = extension

    # Total size of the cache file.
    self.__total_cache_size = 0
    # Total space used within the cache file.
    self.__total_space_used = 0
    self.__max_size = max_size

    self.__location = location
    if not os.path.exists(self.__location):
      os.mkdir(self.__location)

    # Maps IDs of images to their offsets and sizes in the cache file.
    self.__img_ids = {}
    # Maps offsets in the cache file to IDs of images that are there.
    self.__offsets = {}
    # The current location of free space within the cache file.
    self.__free_start = 0
    self.__free_end = 0
    self.__update_free_counter()

    self.__load_existing_cache()

  def __del__(self):
    """ Cleans up the cache data.
    NOTE: This is the reason why you DO NOT want to use SIGKILL on a process
    with a DiskCache open. It will almost certainly corrupt the cache. """
    logger.info("DiskCache: Cleaning up...")

    # Close the data file.
    self.__data_file.close()

    # Write out the data map file.
    cache_map_location = os.path.join(self.__location, "cache_map.pkl")
    map_file = open(cache_map_location, "wb")
    pickle.dump((self.__img_ids, self.__offsets, self.__free_start,
                 self.__free_end), map_file)
    map_file.close()

  def __update_free_counter(self):
    """ Updates the size of the free portion of the file. """
    free_size = self.__free_end - self.__free_start
    if free_size < 0:
      # One has wrapped all the way.
      free_size = self.__free_end + (self.__total_cache_size - \
                                     self.__free_start)

    self.__total_space_used = self.__total_cache_size - free_size
    logger.debug("Disk cache space used: %d" % (self.__total_space_used))

  def __add_image_data(self, size):
    """ Updates the size of the free portion of the file when a new image is
    added.
    Args:
      The size of the image we are adding. """
    self.__free_start += size
    self.__free_start %= self.__total_cache_size

    self.__total_space_used += size
    if self.__total_space_used > self.__total_cache_size:
      logger.critical("Using %d bytes in a cache of size %d!" % \
                      (self.__total_space_used, self.__total_cache_size))
      raise ValueError("Not enough space to add %d bytes in free portion." % \
                       (size))

  def __remove_image_data(self, size):
    """ Updates the size of the free portion of the file when an image is
    removed.
    Args:
      The size of the image we are removing. """
    self.__free_end += size
    self.__free_end %= self.__total_cache_size

    self.__total_space_used -= size
    if self.__total_space_used < 0:
      logger.critical("Cannot remove %d bytes from empty cache." % (size))
      raise ValueError("Cannot remove %d bytes from empty cache." % (size))

  def __load_existing_cache(self):
    """ Accounts for any data that already exists in the cache. """
    # Load the cache map file, which will tell us where everything is in the
    # cache.
    cache_map_location = os.path.join(self.__location, "cache_map.pkl")
    if os.path.exists(cache_map_location):
      logger.debug("Loading %s..." % (cache_map_location))
      cache_map_file = file(cache_map_location, "rb")
      self.__img_ids, self.__offsets, self.__free_start, self.__free_end \
          = pickle.load(cache_map_file)
      cache_map_file.close()

      logger.debug("Free start, free end: %d, %d" % (self.__free_start,
                                                     self.__free_end))

    cache_data_location = os.path.join(self.__location, "cache_data.dat")
    if not os.path.exists(cache_map_location):
      # Make the data file if it doesn't exist already.
      logger.info("Creating new cache data file: %s" % (cache_data_location))
      data_file = open(cache_data_location, "wb")
      data_file.close()

    # Load the cache data file, for reading and appending.
    logger.debug("Loading %s..." % (cache_data_location))
    self.__data_file = file(cache_data_location, "r+b")

    self.__total_cache_size = os.stat(cache_data_location).st_size
    logger.info("Total cache size: %d", self.__total_cache_size)

    self.__update_free_counter()

  def __maintain_size(self):
    """ Makes sure we stay within the bounds of the cache size limit. If we're
    over, it deletes the oldest files until we're at a better size. """
    # If we have no maximum size, we don't have to do anything.
    if self.__max_size == None:
      return

    while self.__total_cache_size > self.__max_size:
      # For now, we're just going to evict by time of addition, because it's
      # easier.
      # TODO (danielp): Implement better eviction policy.
      self.evict_next_image()

  def evict_next_image(self):
    """ Removes the next image from the cache. """
    # Find the image to remove.
    image_offset = self.__free_end
    wnid = self.__offsets[image_offset]
    logger.debug("Removing image: %s" % (wnid))
    label, number = utils.split_img_id(wnid)

    # Decrease cache size.
    _, image_size = self.__img_ids[wnid]

    # Remove it from various data structures.
    self.__img_ids.pop(wnid)
    self.__offsets.pop(image_offset)

    # Update the free section counters.
    self.__remove_image_data(image_size)

  def add(self, image, name, label):
    """ Adds a new image to the cache. If the label is not known, it
    automatically adds that too.
    Args:
      image: The image data to add.
      name: The name of the image. Should be unique within the label.
      label: The image label. """
    img_id = utils.make_img_id(label, name)
    if img_id in self.__img_ids:
      raise ValueError("Attempt to add duplicate image %s." % \
                       img_id)

    # Compress the image for storage.
    compressed = cv2.imencode(self.__extension, image)[1]

    # Check if we have enough free space to write data there.
    free_space = self.__total_cache_size - self.__total_space_used
    wrote_at = 0
    if free_space >= len(compressed):
      logger.debug("Putting image of size %d in space of size %d." % \
                   (len(compressed), free_space))

      # Write the image at the appropriate place in the file.
      self.__data_file.seek(self.__free_start)
      self.__data_file.write(compressed)

      wrote_at = self.__free_start

      # Update the free space counter.
      self.__add_image_data(len(compressed))

    else:
      logger.debug("Saving image %s at end of file." % \
                   utils.make_img_id(label, name))

      # Add it to the end of the file.
      self.__data_file.seek(0, 2)
      self.__data_file.write(compressed)

      wrote_at = self.__total_cache_size

      # Now the file just got bigger.
      self.__total_cache_size += len(compressed)
      self.__total_space_used += len(compressed)

    logger.debug("Wrote image at offset %d." % (wrote_at))
    self.__img_ids[img_id] = (wrote_at, len(compressed))
    self.__offsets[wrote_at] = img_id

    # Make sure we stay within the size constraint.
    self.__maintain_size()

  def __do_get(self, label, name):
    """ Gets an image from the cache. It does not check if an image is actually
    in the cache.
    Args:
      label: The label of the image.
      name: The name of the image.
    Returns: The image data, or None if the image (or label) doesn't exist in
             the cache. """
    # Get the image offset and size.
    img_id = utils.make_img_id(label, name)
    offset, size = self.__img_ids[img_id]

    # Read and decode the image data.
    self.__data_file.seek(offset)
    raw_data = self.__data_file.read(size)
    file_bytes = np.asarray(bytearray(raw_data), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    return image

  def is_in_cache(self, label, name):
    """ Quickly tests if an image is in the cache.
    Args:
      label: The label of the image to check.
      name: The name of the image to check.
    Returns:
      True if the image is in the cache, False otherwise. """
    img_id = utils.make_img_id(label, name)
    return img_id in self.__img_ids

  def get(self, synset, name):
    """ Gets an image from the cache.
    Args:
      label: The label it belongs to.
      name: The image name. It should be unique within the label.
    Returns: The image data, or None if the image (or label) doesn't exist in
             the cache. """
    # Flush any pending writes to the file before we try to read.
    self.__data_file.flush()

    if not self.is_in_cache(synset, name):
      return None
    return self.__do_get(synset, name)

  def bulk_get(self, image_pairs):
    """ Gets a large number of images from the cache at once.
    For a large batch of images, this can be a lot faster than running get
    individually.
    Args:
      image_pairs: Set of (synset, name) pairs for each image to load.
    Returns:
      A dictionary mapping the unique IDs of loaded images to the actual image
      data, and a list of the unique IDs of images that were not found. """
    def compare_offsets(image1_pair, image2_pair):
      """ Compares the offsets of two images in the file.
      Args:
        image1_pair: The (label, name) pair of the first image.
        image2_pair: The (label, name) pair of the second image.
      Returns:
        True if image1 has a smaller offset than image2, False otherwise. """
      label1, name1 = image1_pair
      label2, name2 = image2_pair
      img_id1 = utils.make_img_id(label1, name1)
      img_id2 = utils.make_img_id(label2, name2)

      offset1, _ = self.__img_ids[img_id1]
      offset2, _ = self.__img_ids[img_id2]

      return offset1 < offset2

    # Flush any pending writes to the file before we try to read.
    self.__data_file.flush()

    # Process images that don't exist.
    not_found = []
    to_remove = []
    for label, name in image_pairs:
      img_id = utils.make_img_id(label, name)
      if not self.is_in_cache(label, name):
        not_found.append(img_id)
        to_remove.append((label, name))

    for pair in to_remove:
      image_pairs.remove(pair)

    # Sort the images in the order of their offsets in the file. This is so the
    # needle on the disk doesn't have to jump around everywhere.
    sorted_pairs = sorted(image_pairs, cmp=compare_offsets)

    # Now go and load everything in that order.
    loaded = {}
    for label, name in sorted_pairs:
      image = self.__do_get(label, name)
      assert image is not None
      img_id = utils.make_img_id(label, name)
      loaded[img_id] = [image]

    return loaded, not_found

  def get_sequential(self, start_label, start_name, number_of_images,
                     use_only=None):
    """ Gets a number of images that are stored sequentially in the cache. This
    is the fastest way to load a large number of images from the cache.
    Args:
      start_label: The label of the image to start form.
      start_name: The name of the image to start from.
      number_of_images: Total number of images to load, including the start one.
      use_only: Specifies a set of images. If a loaded image is not in that set,
                it will not be included. Images are specified by image ID.
    Returns:
      A dictionary mapping unique IDs of loaded images to the actual image data.
      It may not contain all the images requested, if there is not sufficient
      data in the cache. It will also return None if the first image is not in
      the cache. Furthermore, it returns the image ID to start at if loading the
      sequential batch directly after this one. This will be None if it hits the
      end of the cache. """
    if not self.is_in_cache(start_label, start_name):
      return None

    # Flush any pending writes to the file before we try to read.
    self.__data_file.flush()

    images = {}

    # Get the size of the data file.
    cache_data_location = os.path.join(self.__location, "cache_data.dat")
    total_size = os.path.getsize(cache_data_location)

    # Compute the offsets for the chunk of the file we need to read.
    start_img_id = utils.make_img_id(start_label, start_name)
    start_offset, size = self.__img_ids[start_img_id]
    end_offset = start_offset
    num_loaded = 0
    to_load = []

    num_rejected = 0
    while (num_loaded < number_of_images + 1 and end_offset < total_size):
      # Find the next image.
      if end_offset not in self.__offsets:
        raise RuntimeError("Unexpected free space in cache. Please repair the \
                            cache.")
      img_id = self.__offsets[end_offset]
      end_offset, size = self.__img_ids[img_id]

      old_end_offset = end_offset
      end_offset = end_offset + size
      if end_offset == self.__free_start:
        # We're in the free space. Just skip over this.
        end_offset = self.__free_end
      elif (end_offset > self.__free_start and end_offset < self.__free_end):
        raise RuntimeError("Data overlaps free space in cache. Please repair \
                            the cache.")

      # We want to account for any free space we skipped here when we add the
      # current image.
      to_load.append((img_id, end_offset - old_end_offset))

      # If we're not going to use it, we still want to add it to the list, since
      # we have to read images in a chain. However, it doesn't count towards the
      # total number of images loaded.
      if (use_only == None or img_id in use_only):
        num_loaded += 1
      else:
        num_rejected += 1
        if num_rejected > number_of_images:
          # At this point, our loading is going to be very inneficient and
          # possibly require an unfeasible amount of memory, so we might as well
          # give up.
          logger.debug("Got too many bad images, giving up.")
          break

    next_image = None
    if num_loaded > number_of_images:
      # We know the next image to start at.
      next_image, _ = to_load.pop()
      logger.debug("First image of next sequential batch: %s" % (next_image))

    # Read that part of the file.
    logger.debug("Reading file from %d to %d." % (start_offset, end_offset))
    self.__data_file.seek(start_offset)
    raw_data = self.__data_file.read(end_offset - start_offset)

    # Load the actual image data.
    raw_data = np.asarray(bytearray(raw_data), dtype=np.uint8)
    for img_id, size in to_load:
      if (use_only == None or img_id in use_only):
        # OpenCV knows when the image ends, so we don't have to worry about bounding
        # our slice on one side.
        image = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
        images[img_id] = [image]
      else:
        logger.debug("Skipping image: %s" % (img_id))

      raw_data = raw_data[size:]

    return (images, next_image)

  def get_first_in_cache(self, use_only=None):
    """ Gets the image id of the first image in the cache.
    Args:
      use_only: If specified, it will look for the earliest out of that subset
      of images.
    Returns:
      The image ID of the first image in the cache. """
    if not use_only:
      # Get the first image in the cache, period.
      if 0 not in self.__offsets:
        if not self.__free_start and self.__free_end:
          # A special case is when we have a free segment starting at zero. This
          # is the only case in which this situation is valid.
          return self.__offsets[self.__free_end]
        raise ValueError("Expected image at offset 0, but found none. Try \
                          repairing cache.")

      return self.__offsets[0]

    # Otherwise, we'll have to go through everything in use_only.
    min_offset = sys.maxint
    first_image = None
    for img_id in use_only:
      offset, _ = self.__img_ids[img_id]

      if offset < min_offset:
        min_offset = offset
        first_image = img_id

    return first_image

  def get_cache_size(self):
    """
    Returns:
      The size of the file used to store the cached data. """
    return self.__total_cache_size

class MemoryBuffer(Cache):
  """ Set of images stored contiguously in memory. This is designed so that it
  can be used as a staging area before a batch is transferred into GPU memory.
  """

  def __init__(self, image_shape, batch_size, num_batches, channels=1,
               num_patches=1):
    """
    Args:
      image_shape: Two element tuple containing the x and y dimensions of the
                   image.
      batch_size: The size of each batch of images.
      num_batches: The number of batches we should be able to hold.
      channels: Number of channels the images have.
      num_patches: The number of patches that are expected for each image.
    """
    super(MemoryBuffer, self).__init__()

    self.__image_shape = image_shape
    self.__batch_size = batch_size
    self.__num_batches = num_batches
    self.__num_patches = num_patches

    self.__num_images = self.__batch_size * self.__num_batches * \
                        self.__num_patches
    logger.debug("Creating mem buffer with total capacity %d." % \
                 (self.__num_images))

    # This will be are underlying storage for the cache.
    self.__channels = channels
    image_x, image_y = self.__image_shape
    shape = (self.__num_images, image_x, image_y, self.__channels)
    logger.debug("MemBuffer: Creating array of shape: %s" % (str(shape)))
    self.__storage = np.empty(shape, dtype="uint8")

    self.__fill_index = 0
    # Fill index for the label array.
    self.__label_fill_index = 0
    # Maps image names to indices in the underlying array.
    self.__image_indices = {}
    # Keeps a list of image labels in the order that they were added.
    self.__labels = [None] * self.__batch_size * self.__num_batches
    # Keeps a list of image names in the order they were added.
    self.__names = [None] * self.__batch_size * self.__num_batches

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
      raise ValueError("Not enough data in buffer for a complete batch.""")

  def __convert_image_for_buffer(self, image):
    """ Converts a raw image as loaded by OpenCV into a form that is compatible
    with the buffer.
    Args:
      image: The image to convert.
    Returns:
      The converted version of the image. """
    if (len(image.shape) != 2 and len(image.shape) != 3):
      raise ValueError("Expected image of 2 or 3 dimensions.")

    if len(image.shape) == 2:
      # If we have a monochrome image, it's not going to have a dimension for
      # the channels, so we have to add it.
      image = np.expand_dims(image, 2)

    return image

  def add(self, image, name, label):
    """ Adds a new image to the buffer.
    Args:
      image: The image data to add.
      name: The name of the image. Should be unique within the label.
      label: The label of the image. """
    self.__storage[self.__fill_index] = self.__convert_image_for_buffer(image)

    unique_identifier = utils.make_img_id(label, name)
    self.__image_indices[unique_identifier] = self.__fill_index

    self.__labels[self.__label_fill_index] = label
    self.__names[self.__label_fill_index] = name

    self.__increment_fill_index()

  def add_patches(self, patches, name, label):
    """ Similar to add, except that it adds multiple patches for the same image.
    Args:
      patches: The list of patches to store.
      name: The name of the image. Should be unique within the label.
      label: The label of the image. """
    if len(patches) != self.__num_patches:
      raise ValueError("Expected %d patches, got %d." % (self.__num_patches,
                                                         len(patches)))

    patch_locations = []
    for i, patch in enumerate(patches):
      location = self.__fill_index + i * self.__batch_size
      self.__storage[location] = self.__convert_image_for_buffer(patch)
      patch_locations.append(location)


    unique_identifier = utils.make_img_id(label, name)
    self.__image_indices[unique_identifier] = patch_locations

    self.__labels[self.__label_fill_index] = label
    self.__names[self.__label_fill_index] = name

    self.__increment_fill_index()

  def get(self, label, name):
    """ Gets an image that was added to the buffer. If there are multiple
    patches of this image, it returns a list of all the patches.
    Args:
      label: The label of the image.
      name: The name of the image.
    Returns:
      The image data. """
    unique_identifier = utils.make_img_id(label, name)
    index = self.__image_indices[unique_identifier]

    if type(index) is list:
      # There are multiple patches.
      patches = []
      for i in index:
        patches.append(self.__storage[index])
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
      The batch data, and label data, and the image name data. """
    end_index = self.__batch_index + self.__batch_size * self.__num_patches
    logger.debug("Getting batch %d:%d." % (self.__batch_index, end_index))
    batch = self.__storage[self.__batch_index:end_index]

    # We only have one label, even if we have multiple patches.
    end_labels = self.__label_batch_index + self.__batch_size
    logger.debug("Getting batch labels: %d:%d." % (self.__label_batch_index,
                                                   end_labels))
    labels = self.__labels[self.__label_batch_index:end_labels]
    names = self.__names[self.__label_batch_index:end_labels]

    self.__increment_batch_index()

    return batch, labels, names

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
    return (self.__data_in_buffer / self.__num_patches)

  def space_remaining(self):
    """
    Returns:
      The total number of images that can still be added to this buffer before
      it's full. """
    return (self.get_max_patches() - self.__data_in_buffer) / self.__num_patches
