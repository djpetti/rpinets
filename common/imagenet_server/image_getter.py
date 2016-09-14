import json
import logging
import os
import random
import time
import urllib2

from randomset import RandomSet
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

def _write_url_file(images, separator=" "):
  """ ImageNet generally uses a specific format to store the URLs of images.
  This converts a set of images into this format.
  Args:
    images: The set of images to write. It should contain tuples with image
    WNIDs and URLs.
    separator: Default string used to separate WNID from URL in the file.
  Returns:
    A raw string which can then be written to a file. """
  file_data = ""
  for wnid, url in images:
    line = "%s%s%s\n" % (wnid, separator, url)
    file_data += line
  return file_data


class ImageGetter(object):
  """ Gets random sets of images for use in training and testing. """

  def __init__(self, synset_location, cache_location, batch_size,
               preload_batches=1, test_percentage=0.1, download_words=False):
    """
    Args:
      synset_location: Where to save synsets. Will be created if it doesn't
      exist.
      cache_location: Where to cache downloaded images. Will be created if it
      doesn't exist.
      batch_size: The size of each batch to load.
      preload_batches: The number of batches that will be preloaded. Increasing
      this number uses more RAM, but can greatly increase performance.
      test_percentage: The percentage of the total images that will be used for
      testing.
      download_words: Whether to download the words for each synset as well as
      just the numbers. """
    self._cache = cache.DiskCache(cache_location, 50000000000,
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
    self._train_set = _TrainingDataset(train, self._cache, self.__batch_size,
                                       preload_batches=preload_batches)
    self._test_set = _TestingDataset(test, self._cache, self.__batch_size,
                                     preload_batches=preload_batches)

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

  def __remove_bad_images(self, bad_images):
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
    images, failures = self._train_set.get_random_batch()

    # Remove failures from the synset files.
    self.__remove_bad_images(failures)

    return images

  def get_random_test_batch(self):
    """ Gets a random testing batch.
    Returns:
      The array of loaded images, and the list of labels. """
    images, failures = self._test_set.get_random_batch()

    # Remove failures from the synset files.
    self.__remove_bad_images(failures)

    return images


class FilteredImageGetter(ImageGetter):
  """ Works like an ImageGetter, but only loads images from a specific
  pre-defined file containing image names and their URLs. """

  def __init__(self, url_file, cache_location, batch_size, remove_bad=True,
               **kwargs):
    """
    Args:
      url_file: Name of the file containing the images to use.
      cache_location: Where to store downloaded images.
      batch_size: The size of each batch.
      remove_bad: Whether to modify url_file in order to remove broken links.
      The default is True. """
    self.__url_file = url_file
    self.__remove_bad = remove_bad

    super(FilteredImageGetter, self).__init__(None, cache_location, batch_size, **kwargs)

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

  def __write_url_file(self):
    """ Writes the current images in this dataset to a URL file. """
    logger.debug("Writing images to URL file '%s'..." % (self.__url_file))

    train_images = self._train_set.get_images()
    test_images = self._test_set.get_images()
    combined_images = train_images.union(test_images)

    file_data = _write_url_file(combined_images, separator="\t")

    url_file = open(self.__url_file, "w")
    url_file.write(file_data)
    url_file.close()

  def get_random_train_batch(self):
    """ Gets a random training batch.
    Returns:
      The array of loaded images, and the list of labels. """
    images, failures = self._train_set.get_random_batch()

    if self.__remove_bad:
      self.__write_url_file()

    return images

  def get_random_test_batch(self):
    """ Gets a random testing batch.
    Returns:
      The array of loaded images, and the list of labels. """
    images, failures = self._test_set.get_random_batch()

    if self.__remove_bad:
      self.__write_url_file()

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
    self.__images = RandomSet(images)
    self._cache = disk_cache

    self._batch_size = batch_size

    # The number of images that we've currently requested be downloaded.
    self.__num_downloading = 0

    logger.info("Have %d total images in database." % (len(self.__images)))

  def __pick_random_image(self):
    """ Picks a random image from our database.
    Returns:
      wnid and url, and key of the image in the images map. """
    # Pick a random image.
    wnid, url = self.__images.get_random()

    if wnid in self.__already_picked:
      # This is a duplicate, pick another.
      return self.__pick_random_image()
    self.__already_picked.add(wnid)

    return wnid, url

  def get_random_batch(self):
    """ Loads a random batch of images from the whole dataset.
    Returns:
      The array of loaded images, and a list of the synsets each image belongs
      to, as well as a list of all images that failed to download. """
    logger.info("Getting batch.")

    # Keeps track of images that were already used this batch.
    self.__already_picked = set([])

    # Try to download initial images.
    need_images = self._mem_buffer.get_max_images() - self.__num_downloading
    logger.debug("Need to start downloading %d additional images." %
                 (need_images))
    for _ in range(0, need_images):
      self.__load_random_image()
    self.__num_downloading += need_images

    # Wait for 1 batch worth of the downloads to complete,
    # replacing any ones that fail.
    to_remove = []
    successfully_downloaded = 0
    while successfully_downloaded < self._batch_size:
      failures = self._download_manager.get_failures()
      if len(failures):
        logger.debug("Replacing %d failed downloads." % (len(failures)))

      # Remove failed images.
      for synset, name, url in failures:
        # Remove failed image.
        wnid = "%s_%s" % (synset, name)

        logger.info("Removing bad image %s, %s" % (wnid, url))

        # Because we keep running downloader processes in the background through
        # multiple calls to get_random_batch, it's possible that we could try to
        # remove something twice.
        if (wnid, url) in self.__images:
          self.__images.remove((wnid, url))
          to_remove.append((wnid, url))
        else:
          logger.debug("Item already removed: %s" % (wnid))

        # Load a new image to replace it.
        self.__load_random_image()

      downloaded = self._download_manager.update()
      successfully_downloaded += downloaded

      time.sleep(0.2)

    self.__num_downloading -= self._batch_size
    return self._mem_buffer.get_batch(), to_remove

  def __load_random_image(self):
    """ Loads a random image from either the cache or the internet. If loading
    from the internet, it adds it to the download manager, otherwise, it adds
    them to the memory buffer. """
    wnid, url = self.__pick_random_image()
    synset, number = wnid.split("_")

    image = self.__get_cached_image(synset, number)
    if image is None:
      # We have to download the image instead.
      self._download_manager.download_new(synset, number, url)
    else:
      self._mem_buffer.add(image, number, synset)

  def __get_cached_image(self, synset, image_number):
    """ Checks if an image is in the cache, and returns it.
    Args:
      synset: The synset of the image.
      image_number: The image number in the synset.
    Returns:
      The actual image data, or None if the image is not in the cache. """
    # First check in the cache for the image.
    cached_image = self._cache.get(synset, image_number)
    if cached_image is not None:
      # We had a cached copy, so we're done.
      logger.debug("Found image in cache: %s_%s" % (synset, image_number))

      # Select a patch.
      patches = data_augmentation.extract_patches(cached_image)
      cached_image = patches[random.randint(0, len(patches) - 1)]

      return cached_image

    return None

  def get_images(self):
    """ Gets all the images in the dataset. """
    return self.__images


class _TrainingDataset(_Dataset):
  """ A standard training dataset. """


  def __init__(self, images, disk_cache, batch_size, preload_batches=1):
    """ See documentation for superclass method.
    Extra Args:
      preload_batches: How many additional batches to preload. This can greatly
                       increase performace, but uses additional RAM. """
    super(_TrainingDataset, self).__init__(images, disk_cache, batch_size)

    self._mem_buffer = cache.MemoryBuffer(224, batch_size, preload_batches,
                                          channels=3)
    self._download_manager = downloader.DownloadManager(self._cache,
                                                        self._mem_buffer)


class _TestingDataset(_Dataset):
  """ Dataset specifically for testing. One main difference is that it extracts
  every patch for each image, and stores 10 versions of each batch, one for each
  patch type. """

  def __init__(self, images, disk_cache, batch_size, preload_batches=1):
    """ See documentation for _Dataset __init__ function.
    NOTE: batch_size here represents the base batch size. When you request a
    batch, it will actually return 10 times this many images, since it will use
    all the patches.
    Extra Args:
      preload_batches: How many additional batches to preload. This can greatly
                       increase performace, but uses additional RAM. """
    super(_TestingDataset, self).__init__(images, disk_cache, batch_size)

    # We need 10x the buffer space to store the extra patches.
    self._mem_buffer = cache.MemoryBuffer(224, batch_size, preload_batches,
                                          channels=3, num_patches=10)
    self._download_manager = downloader.DownloadManager(self._cache,
                                                        self._mem_buffer,
                                                        all_patches=True)
