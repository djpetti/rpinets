import json
import logging
import os
import random
import time
import urllib2

import cv2

import cache
import data_augmentation
import downloader


logger = logging.getLogger(__name__)


def _parse_url_file(file_data, separator=" ", get_utf=False):
  """ ImageNet generally uses a specific format to store the URLs of images.
  This function parses data in that format.
  Args:
    file_data: The input data to parse.
    separator: Default string used to separate WNID from URL in the file.
    get_utf: If this is True, it will, in addition, return a version of the same
    output that is encoded in UTF-8, which is more efficient if we're dumping to
    a JSON file, since we need to encode it in this format anyway.
  Returns:
    A list of tuples, where each tuple contains a wnid of an image, and the URL
    of that image. """
  lines = file_data.split("\n")[:-1]
  # Split ids and urls.
  mappings = []
  utf_mappings = []
  for line in lines:
    if not line.startswith("n"):
      # Bad line.
      continue

    wnid, url = line.split(separator, 1)
    try:
      unicode_url = url.decode("utf8")
    except UnicodeDecodeError:
      # We got a URL that's not valid unicode. (It does happen.)
      logger.warning("Skipping invalid URL: %s", url)
      continue

    mappings.append([wnid, url])
    if get_utf:
      utf_mappings.append([wnid, unicode_url])

  if get_utf:
    return mappings, utf_mappings
  return mappings


class ImageGetter(object):
  """ Gets random sets of images for use in training and testing. """

  def __init__(self, synset_location, batch_size, test_percentage=0.1,
               download_words=False):
    """
    Args:
      synset_location: Where to save synsets. Will be created if it doesn't
      exist.
      batch_size: The size of each batch to load.
      test_percentage: The percentage of the total images that will be used for
      testing.
      download_words: Whether to download the words for each synset as well as
      just the numbers. """
    self.__cache = cache.DiskCache("image_cache", 50000000000,
                                   download_words=download_words)
    self.__batch_size = batch_size

    self.__synset_location = synset_location

    self._synsets = {}
    self._populate_synsets()

    # Calculate and store sizes for each synset.
    self.__synset_sizes = {}
    self.__total_images = 0
    for synset, urls in self._synsets.iteritems():
      self.__synset_sizes[synset] = len(urls)
      self.__total_images += len(urls)

    # Make internal datasets for training and testing.
    train, test = self.__split_train_test_images(test_percentage)
    self.__train_set = _Dataset(train, self.__cache, self.__batch_size)
    self.__test_set = _Dataset(test, self.__cache, self.__batch_size)

  def _populate_synsets(self):
    """ Populates the synset dictionary. """
    if not os.path.exists(self.__synset_location):
      os.mkdir(self.__synset_location)

    loaded = self.__load_synsets()
    self.__get_image_list(loaded)

  def __split_train_test_images(self, test_percentage):
    """ Chooses the images to use in the training set and testing set.
    NOTE: For memory reasons, it also removes the images from the synset map.
    Args:
      test_percentage: The percentage of the total number of images that are
      testing images.
    Returns:
      The list of training images, and the list of testing images. """
    train = []
    test = []

    for synset in self._synsets.keys():
      for image in self._synsets[synset]:
        rand = random.random()
        if rand < test_percentage:
          test.append(tuple(image))
        else:
          train.append(tuple(image))

      # Remove the synset.
      self._synsets.pop(synset)

    return train, test

  def __get_image_list(self, loaded):
    """ Downloads a comprehensive list of all images available.
    Args:
      loaded: Set of synsets that were already loaded successfully. """
    logger.info("Downloading list of synsets...")
    url = "http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list"
    response = urllib2.urlopen(url, timeout=1000)
    synsets = set(response.read().split("\n")[:-1])

    # Remove any that were already loaded.
    synsets = synsets - loaded
    logger.info("Downloading %d synsets." % (len(synsets)))
    logger.debug("Downloading synsets: %s", synsets)

    # Get the urls for each synset.
    base_url = \
        "http://www.image-net.org/api/text/" \
        "imagenet.synset.geturls.getmapping?wnid=%s"
    for synset in synsets:
      if not synset:
        continue

      logger.info("Downloading urls for synset: %s", synset)
      response = urllib2.urlopen(base_url % (synset), timeout=1000)

      # Parse the image data.
      mappings, utf_mappings = _parse_url_file(response.read(), get_utf=True)

      self._synsets[synset] = mappings
      # Save it for later.
      self.__save_synset(synset, utf_mappings)

  def condense_loaded_synsets(self):
    """ Using is-a relationships obtained from ImageNet, it combines smaller
    synsets until only ones that have a sufficient amount of data remain. """
    logger.info("Condensing synsets...")

    # First, download the list of is-a relationships.
    url = "http://www.image-net.org/archive/wordnet.is_a.txt"
    response = urllib2.urlopen(url)
    lines = response.read().split("\n")[:-1]

    # Convert to actual mappings. Here, the keys are a synset, and the values
    # are what that synset is.
    mappings = {}
    for line in lines:
      if not line:
        # Bad line.
        continue
      sup, sub = line.split()
      mappings[sub] = sup

    # Now, go through our synsets and combine any that are too small.
    while True:
      sizes_to_remove = []
      for synset, size in self.__synset_sizes.iteritems():
        merged_at_least_one = False
        if size < self.MINIMUM_SYNSET_SIZE:
          if synset in mappings:
            # Combine with superset.
            superset = mappings[synset]
            if superset not in self.__synsets:
              logger.warning("Superset %s is not loaded!" % (superset))
            else:
              logger.info("Merging %s with %s." % (synset, superset))

              # It looks like synsets don't by default include sub-synsets.
              self.__synsets[superset].extend(self.__synsets[synset])
              self.__synset_sizes[superset] += self.__synset_sizes[synset]
              logger.debug("New superset size: %d." % \
                          (self.__synset_sizes[superset]))

          # Delete it since it's too small.
          logger.info("Deleting %s with size %d." % (synset, size))
          self.__synsets.pop(synset)
          # We can't pop items from the dictionary while we're iterating.
          sizes_to_remove.append(synset)

          merged_at_least_one = True

      for synset in sizes_to_remove:
        self.__synset_sizes.pop(synset)

      if not merged_at_least_one:
        # We're done here.
        logger.info("Finished merging synsets.")
        break

  def __save_synset(self, synset, data):
    """ Saves a synset to a file.
    Args:
      synset: The name of the synset.
      data: The data that will be saved in the file. """
    if not self.__synset_location:
      # We're not saving synsets.
      return

    logger.info("Saving synset: %s" % (synset))

    synset_path = os.path.join(self.__synset_location, "%s.json" % (synset))
    synset_file = open(synset_path, "w")
    json.dump(data, synset_file)
    synset_file.close()

  def __load_synsets(self):
    """ Loads synsets from a file.
    Returns:
      A set of the synsets that were loaded successfully. """
    loaded = set([])
    loaded_text = []
    logger.info("Loading synsets...")

    for path in os.listdir(self.__synset_location):
      if path.endswith(".json"):
        # Load synset from file.
        synset_name = path[:-5]
        logger.debug("Loading synset %s." % (synset_name))
        full_path = os.path.join(self.__synset_location, path)

        synset_file = file(full_path)
        loaded_text.append((synset_name, synset_file.read()))
        synset_file.close()

    logger.info("De-jsoning...")
    for synset_name, text in loaded_text:
      # Storing stuff in the default unicode format takes up a
      # massive amount of memory.
      self._synsets[synset_name] = \
          [[elem[0].encode("utf8"), elem[1].encode("utf8")] \
            for elem in json.loads(text)]

      loaded.add(synset_name)

    return loaded

  def _remove_bad_images(self, bad_images):
    """ Removes bad images from the synset files.
    Args:
      bad_images: A list of the images to remove. Should contain tuples of the
      image WNID and URL. """
    for wnid, url in bad_images:
      synset, _ = wnid.split("_")

      # Load the proper file.
      file_name = "%s.json" % (synset)
      file_path = os.path.join(self.__synset_location, file_name)
      file_object = file(file_path)
      synset_data = json.load(file_object)
      file_object.close()

      # Remove the correct entries.
      synset_data.remove([wnid, url])

      # Write back to the file.
      self.__save_synset(synset, synset_data)

  def get_synsets(self):
    return self._synsets

  def get_random_train_batch(self):
    """ Gets a random training batch.
    Returns:
      The array of loaded images, and the list of labels. """
    images, failures = self.__train_set.get_random_batch()

    # Remove failures from the synset files.
    self._remove_bad_images(failures)

    return images

  def get_random_test_batch(self):
    """ Gets a random testing batch.
    Returns:
      The array of loaded images, and the list of labels. """
    images, failures = self.__test_set.get_random_batch()

    # Remove failures from the synset files.
    self._remove_bad_images(failures)

    return images


class _Dataset(object):
  """ Represents a single dataset, with a fixed group of images that it draws
  randomly from. """

  def __init__(self, images, disk_cache, batch_size):
    """
    Args:
      images: The list of images that makes up this dataset. The list should
      contain tuples with each image's WNID and URL.
      disk_cache: The disk cache where we can store downloaded images.
      batch_size: The size of each batch from this dataset. """
    self.__images = dict(zip(range(0, len(images) - 1), images))
    self.__cache = disk_cache

    self.__mem_buffer = cache.MemoryBuffer(224, batch_size, channels=3)
    self.__download_manager = downloader.DownloadManager(200,
        self.__cache, self.__mem_buffer)
    self.__batch_size = batch_size

    logger.info("Have %d total images in database." % (len(self.__images)))

  def __pick_random_image(self):
    """ Picks a random image from our database.
    Returns:
      wnid and url, and key of the image in the images map. """
    # Pick a random image.
    key = random.randint(0, len(self.__images) - 1)
    wnid, url = self.__images[key]

    if wnid in self.__already_picked:
      # This is a duplicate, pick another.
      return self.__pick_random_image()
    self.__already_picked.add(wnid)

    return wnid, url, key

  def get_random_batch(self):
    """ Loads a random batch of images from the whole dataset.
    Returns:
      The array of loaded images, and a list of the synsets each image belongs
      to, as well as a list of all images that failed to download. """
    logger.info("Getting batch.")

    self.__mem_buffer.clear()

    # Keeps track of images that were already used this batch.
    self.__already_picked = set([])

    # Try to download initial images.
    loading = {}
    for _ in range(0, self.__batch_size):
      wnid, key = self.__load_random_image()
      loading[wnid] = key

    # Wait for all the downloads to complete, replacing any ones that fail.
    to_remove = []
    while True:
      failures = self.__download_manager.get_failures()
      if len(failures):
        logger.debug("Replacing %d failed downloads." % (len(failures)))

      # Remove failed images.
      for synset, name, url in failures:
        # Load a new image to replace it.
        wnid, key = self.__load_random_image()
        loading[wnid] = key

        # Remove failed image.
        wnid = "%s_%s" % (synset, name)
        logger.info("Removing bad image %s." % (wnid))
        self.__images.pop(loading[wnid])
        loading.pop(wnid)

        to_remove.append((wnid, url))

      if not self.__download_manager.update():
        break

      time.sleep(0.2)

    return self.__mem_buffer.get_storage(), to_remove

  def __load_random_image(self):
    """ Loads a random image from either the cache or the internet. If loading
    from the internet, it adds it to the download manager, otherwise, it adds
    them to the memory buffer.
    Returns:
      The WNID and key of the loaded image. """
    wnid, url, key = self.__pick_random_image()
    synset, number = wnid.split("_")

    image = self.__get_cached_image(synset, number)
    if image is None:
      # We have to download the image instead.
      self.__download_manager.download_new(synset, number, url)
    else:
      self.__mem_buffer.add(image, number, synset)

    return wnid, key

  def __get_cached_image(self, synset, image_number):
    """ Checks if an image is in the cache, and returns it.
    Args:
      synset: The synset of the image.
      image_number: The image number in the synset.
    Returns:
      The actual image data, or None if the image is not in the cache. """
    # First check in the cache for the image.
    cached_image = self.__cache.get(synset, image_number)
    if cached_image != None:
      # We had a cached copy, so we're done.
      logger.debug("Found image in cache: %s_%s" % (synset, image_number))

      # Select a patch.
      patches = data_augmentation.extract_patches(cached_image)
      cached_image = patches[random.randint(0, len(patches) - 1)]

      return cached_image

    return None


class FilteredImageGetter(ImageGetter):
  """ Works like an ImageGetter, but only loads images from a specific
  pre-defined file containing image names and their URLs. """

  def __init__(self, url_file, batch_size, **kwargs):
    """
    Args:
      url_file: Name of the file containing the images to use.
      batch_size: The size of each batch. """
    self.__url_file = url_file

    super(FilteredImageGetter, self).__init__(None, batch_size, **kwargs)

  def _populate_synsets(self):
    """ Populates the synsets dict. """
    # Load data from the url file.
    urls = file(self.__url_file).read()
    images = _parse_url_file(urls, separator="\t")

    # Separate it by synset and load it into the dictionary.
    for wnid, url in images:
      synset, _ = wnid.split("_")

      if synset not in self._synsets:
        self._synsets[synset] = []
      self._synsets[synset].append([wnid, url])

  def _remove_bad_images(self, *args, **kwargs):
    """ No-op to stop it from trying to access synsets that we don't know about.
    """
    pass
