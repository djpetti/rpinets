import logging
import os
import shutil
import unittest

import cv2

import numpy as np

import cache


logger = logging.getLogger(__name__)


class DiskCacheTest(unittest.TestCase):
  """ Tests for the DiskCache. """
  def setUp(self):
    # Clear any existing cache.
    if os.path.exists("test_cache"):
      logger.debug("Removing old cache directory...")
      shutil.rmtree("test_cache")

    # Create a new testing directory.
    os.mkdir("test_cache")

    # Load testing images.
    this_dir = os.path.dirname(os.path.realpath(__file__))
    image_dir = os.path.join(this_dir, "test_images")
    self.__images = []
    for image_file in os.listdir(image_dir):
      image_path = os.path.join(image_dir, image_file)
      image = cv2.imread(image_path)
      if image is None:
        raise ValueError("Could not load image '%s'." % (image_path))

      self.__images.append(image)

    # Create a DiskCache instance for testing. Use TIFF instead of JPEG for
    # lossless data, so we can compare images going into and out of the cache
    # properly.
    self.__cache = cache.DiskCache("test_cache", 10000000, extension=".tiff")

  def tearDown(self):
    # Run the destructor first so we don't get errors when it tries to write out
    # files.
    try:
      del(self.__cache)
    except AttributeError:
      # The tests might want to force the cache destruction, which is fine.
      pass

    # Clean up the testing cache.
    shutil.rmtree("test_cache")

  def test_single_item(self):
    """ Test that we can store and load a single item. """
    test_image = self.__images[0]

    self.__cache.add(test_image, "image1", "label1")
    got_image = self.__cache.get("label1", "image1")
    self.assertTrue(np.array_equal(test_image, got_image))

  def test_saving(self):
    """ Tests that we can store an item, and it will still be there when we
    reload the cache. """
    test_image = self.__images[0]

    self.__cache.add(test_image, "image1", "label1")
    # Force it to write out to the disk.
    del(self.__cache)

    new_cache = cache.DiskCache("test_cache", 1000000, extension=".tiff")
    got_image = new_cache.get("label1", "image1")
    self.assertTrue(np.array_equal(test_image, got_image))

  def test_multiple_items(self):
    """ Test that we can store and load multiple items. """
    # Store all the images.
    for i, image in enumerate(self.__images):
      name = "image%d" % (i)
      self.__cache.add(image, name, "label1")

    # Load all the images.
    for i, image in enumerate(self.__images):
      name = "image%d" % (i)
      got_image = self.__cache.get("label1", name)
      self.assertTrue(np.array_equal(image, got_image))

  def test_eviction(self):
    """ Test that we can evict images from the cache. """
    # Store all the images.
    for i, image in enumerate(self.__images):
      name = "image%d" % (i)
      self.__cache.add(image, name, "label1")

    # Evict the first image we put in.
    cache_size = self.__cache.get_cache_size()
    self.__cache.evict_next_image()

    # It should no longer show up.
    self.assertIsNone(self.__cache.get("label1", "image0"))
    self.assertEqual(cache_size, self.__cache.get_cache_size())

    # If we add the same image again, it should be exactly overwritten.
    self.__cache.add(self.__images[0], "image0", "label1")
    self.assertEqual(cache_size, self.__cache.get_cache_size())

    # All the images should still be valid.
    for i, image in enumerate(self.__images):
      name = "image%d" % (i)
      got_image = self.__cache.get("label1", name)
      self.assertTrue(np.array_equal(image, got_image))

  def test_multiple_eviction(self):
    """ Test that we can evict every image from the cache, and then write them
    back. """
    # We're going to save out one image for a test at the end.
    images = self.__images[:-1]

    # Store all the images.
    for i, image in enumerate(images):
      name = "image%d" % (i)
      self.__cache.add(image, name, "label1")

    # Evict all the images.
    cache_size = self.__cache.get_cache_size()
    for i in range(0, len(images)):
      self.__cache.evict_next_image()
    self.assertEqual(cache_size, self.__cache.get_cache_size())

    # Add them all back, in a different order.
    images.reverse()
    for i, image in enumerate(images):
      name = "image%d" % (i)
      self.__cache.add(image, name, "label1")

    # Total size still shouldn't have changed.
    self.assertEqual(cache_size, self.__cache.get_cache_size())

    # Now, add a last image, and make sure it expands the cache.
    self.__cache.add(self.__images[-1], "last_image", "label1")
    self.assertGreater(self.__cache.get_cache_size(), cache_size)

    # All the images should still be valid.
    for i, image in enumerate(images):
      name = "image%d" % (i)
      got_image = self.__cache.get("label1", name)
      self.assertTrue(np.array_equal(image, got_image))
    got_image = self.__cache.get("label1", "last_image")
    self.assertTrue(np.array_equal(self.__images[-1], got_image))
