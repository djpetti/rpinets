import logging
import os
import shutil
import unittest

import cv2

import numpy as np

import cache
import utils


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

  def test_bulk_get(self):
    """ Tests that we can use the bulk_get method to load a bunch of data from
    the cache at once. """
    # Store all the images.
    image_pairs = set()
    for i, image in enumerate(self.__images):
      name = "image%d" % (i)
      self.__cache.add(image, name, "synset1")
      image_pairs.add(("synset1", name))

    # Add another image to image_pairs that doesn't exist.
    bad_name = "image%d" % (len(self.__images))
    bad_id = utils.make_img_id("synset1", bad_name)
    image_pairs.add(("synset1", bad_name))

    # Load everything.
    loaded, not_found = self.__cache.bulk_get(image_pairs)

    # Check that it found everything it should.
    for i, image in enumerate(self.__images):
      name = "image%d" % (i)
      img_id = utils.make_img_id("synset1", name)
      self.assertIn(img_id, loaded)
      got_image = loaded[img_id]
      self.assertTrue(np.array_equal(image, got_image))

    # Check that it didn't find the last one.
    self.assertEqual(1, len(not_found))
    self.assertEqual(bad_id, not_found[0])

  def test_get_sequential(self):
    """ Tests that we can use the get_sequential method to load a bunch of
    contiguous data from the cache. """
    # Store all the images.
    for i, image in enumerate(self.__images):
      name = "image%d" % (i)
      self.__cache.add(image, name, "synset1")

    # Try to read everything back from the cache.
    loaded = self.__cache.get_sequential("synset1", "image0",
                                         len(self.__images))

    # Check that it found everything it should.
    for i, image in enumerate(self.__images):
      name = "image%d" % (i)
      img_id = utils.make_img_id("synset1", name)
      self.assertIn(img_id, loaded)
      got_image = loaded[img_id]
      self.assertTrue(np.array_equal(image, got_image))

  def test_sequential_free_space(self):
    """ Tests that get_sequential still works when the cache has free space in
    it. """
    # Store all the images.
    for i, image in enumerate(self.__images):
      name = "image%d" % (i)
      self.__cache.add(image, name, "synset1")

    # Evict an image from the cache.
    self.__cache.evict_next_image()

    # Try to read everything remaining back from the cache.
    loaded = self.__cache.get_sequential("synset1", "image1",
                                         len(self.__images) - 1)

    # Check that it found everything it should.
    for i, image in enumerate(self.__images[1:]):
      name = "image%d" % (i + 1)
      img_id = utils.make_img_id("synset1", name)
      self.assertIn(img_id, loaded)
      got_image = loaded[img_id]
      self.assertTrue(np.array_equal(image, got_image))

  def test_sequential_premature_end(self):
    """ Tests that get_sequential works when we prematurely run out of images to
    read. """
    # Store all the images.
    for i, image in enumerate(self.__images):
      name = "image%d" % (i)
      self.__cache.add(image, name, "synset1")

    # Try to read everything back from the cache, and more.
    loaded = self.__cache.get_sequential("synset1", "image0",
                                         len(self.__images) + 1)

    # Check that it found everything it should.
    for i, image in enumerate(self.__images):
      name = "image%d" % (i)
      img_id = utils.make_img_id("synset1", name)
      self.assertIn(img_id, loaded)
      got_image = loaded[img_id]
      self.assertTrue(np.array_equal(image, got_image))
    # Nothing extra should be there.
    self.assertEqual(len(self.__images), len(loaded))

  def test_sequential_use_only(self):
    """ Tests that get_sequential works when we give it a set of the images it
    can use. """
    # Store all the images.
    use_only = set()
    for i, image in enumerate(self.__images):
      name = "image%d" % (i)
      use_only.add(utils.make_img_id("synset1", name))
      self.__cache.add(image, name, "synset1")

    # Remove the first image from use_only.
    use_only.remove(utils.make_img_id("synset1", "image0"))
    # Try to read everything back from the cache.
    loaded = self.__cache.get_sequential("synset1", "image0",
                                         len(self.__images),
                                         use_only=use_only)

    # Check that it found everything except the first one.
    for i, image in enumerate(self.__images[1:]):
      name = "image%d" % (i + 1)
      img_id = utils.make_img_id("synset1", name)
      self.assertIn(img_id, loaded)
      got_image = loaded[img_id]
      self.assertTrue(np.array_equal(image, got_image))
    self.assertNotIn("synset1_image0", loaded)

  def test_get_first(self):
    """ Tests that get_first_in_cache works properly. """
    # Store all the images.
    use_only = set()
    for i, image in enumerate(self.__images):
      name = "image%d" % (i)
      use_only.add(utils.make_img_id("synset1", name))
      self.__cache.add(image, name, "synset1")
    # Remove the first image from use_only.
    use_only.remove(utils.make_img_id("synset1", "image0"))

    # Get the first one.
    first_image = self.__cache.get_first_in_cache()
    self.assertEqual(utils.make_img_id("synset1", "image0"), first_image)

    # Now, give it a use_only argument.
    first_image = self.__cache.get_first_in_cache(use_only=use_only)
    self.assertEqual(utils.make_img_id("synset1", "image1"), first_image)
