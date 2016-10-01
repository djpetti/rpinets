import cPickle as pickle
import json
import logging
import operator
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
               preload_batches=1, test_percentage=0.1, load_datasets_from=None,
               download_words=False):
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
      load_datasets_from: The common part of the path to the files that we want
      to load the training and testing datasets from. If this is not specified,
      it will create new training and testing sets.
      download_words: Whether to download the words for each synset as well as
      just the numbers. """
    self._cache = cache.DiskCache(cache_location, 50000000000,
                                   download_words=download_words)
    self.__batch_size = batch_size

    self._synset_location = synset_location

    self._synsets = {}
    self._populate_synsets()

    # Calculate and store sizes for each synset.
    self.__synset_sizes = {}
    self.__total_images = 0
    for synset, urls in self._synsets.iteritems():
      self.__synset_sizes[synset] = len(urls)
      self.__total_images += len(urls)

    self.__load_datasets_from = load_datasets_from

    # Make internal datasets for training and testing.
    if (self.__load_datasets_from and \
        (os.path.exists(self.__load_datasets_from + "_training.pkl") and \
         os.path.exists(self.__load_datasets_from + "_testing.pkl"))):
      # Initialize empty datasets.
      self._train_set = _TrainingDataset(set(), self._cache, self.__batch_size,
                                         preload_batches=preload_batches)
      self._test_set = _TestingDataset(set(), self._cache, self.__batch_size,
                                       preload_batches=preload_batches)

      # Use the saved datasets instead of making new ones.
      self.load_datasets(self.__load_datasets_from)

      # We might have deleted stuff since we last saved it, so we want to do
      # some pruning.
      if self._test_set.prune_images(self._synsets):
        logger.info("Updating test database file...")
        self._test_set.save_images(self.__load_datasets_from)
      if self._train_set.prune_images(self._synsets):
        logger.info("Updating train database file...")
        self._train_set.save_images(self.__load_datasets_from)

    else:
      train, test = self.__split_train_test_images(test_percentage)
      self._train_set = _TrainingDataset(train, self._cache, self.__batch_size,
                                         preload_batches=preload_batches)
      self._test_set = _TestingDataset(test, self._cache, self.__batch_size,
                                       preload_batches=preload_batches)

      if self.__load_datasets_from:
        # If we specified a path to load datasets from, make them here for next
        # time.
        logger.warning("Datasets not found. Making new ones and saving in %s." \
                       % (self.__load_datasets_from))
        self.save_datasets(self.__load_datasets_from)

  def _populate_synsets(self):
    """ Populates the synset dictionary. """
    if not os.path.exists(self._synset_location):
      os.mkdir(self._synset_location)

    loaded = self._load_synsets()
    self._get_image_list(loaded)

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

  def _get_image_list(self, loaded):
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

    self._get_synset_urls(synsets)

  def _get_synset_urls(self, synsets):
    """ Get the image URLs for each synset.
    Args:
      synsets: The set of synsets to get URLs for. """
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
      if not mappings:
        raise ValueError("Invalid synset: %s" % (synset))

      self._synsets[synset] = mappings
      # Save it for later.
      self.__save_synset(synset, utf_mappings)

  def condense_loaded_synsets(self, target_categories, minimum_synset_size):
    """ Using is-a relationships obtained from ImageNet, it combines smaller
    synsets until only ones that have a sufficient amount of data remain.
    Args:
      target_categories: The target number of categories that we want.
      minimum_synset_size: The minimum size of a particular synset. """
    def merge_with_superset(synset):
      """ Merges a synset with its superset.
      Returns:
        The name of the superset if the merge was successful, or None if it
        couldn't find a superset. """
      if synset not in mappings:
        return None

      # Combine with superset.
      superset = mappings[synset]

      logger.info("Merging %s with %s." % (synset, superset))

      # It looks like synsets don't by default include sub-synsets.
      self._synsets[superset].extend(self.__synsets[synset])
      self.__synset_sizes[superset] += self.__synset_sizes[synset]
      logger.debug("New superset size: %d." % \
                  (self.__synset_sizes[superset]))

      return superset

    def find_leaves(mappings):
      """ Given a dictionary of is-a relationships, it finds the leaves of the
      hierarchy.
      Args:
        mappings: Dict mapping subsets to supersets.
      Returns:
        Set of all leaf nodes. """
      leaves = set(mappings.keys())
      for superset in mappings.items():
        leaves.discard(superset)

      return leaves

    logger.info("Condensing synsets...")

    # First, download the list of is-a relationships.
    url = "http://www.image-net.org/archive/wordnet.is_a.txt"
    response = urllib2.urlopen(url)
    lines = response.read().split("\n")[:-1]

    # Convert to actual mappings. Here, the keys are a synset, and the values
    # are what that synset is.
    mappings = {}
    reverse_mappings = {}
    for line in lines:
      if not line:
        # Bad line.
        continue
      sup, sub = line.split()
      mappings[sub] = sup
      reverse_mappings[sup] = sub

    # Find the initial leaf nodes.
    leaves = find_leaves(mappings)

    # Now, go through our synsets and combine any that are too small. We start
    # from the leaves and work our way up through the structure.
    while len(leaves):
      leaf = leaves.pop()
      if self.__synset_sizes[leaf] < minimum_synset_size:
        # We need to merge this one.
        superset = merge_with_superset(leaf)

        # Replace the leaf with its parent, since it's the new leaf now.
        if superset:
          leaves.add(superset)
        else:
          # In this case, we want to try merging downwards.
          child = reverse_mappings[leaf]
          merge_with_superset(child)
          # Now we still want to be in the leaf nodes set, in case we need to
          # merge more.
          leaves.add(leaf)
          # We also need to update our mappings.
          grandchild = reverse_mappings[child]
          reverse_mappings[leaf] = grandchild

          # We want to remove the child though, since it got merged.
          leaf = child

        # The only way the merge failed is if leaf was actually a root node.
        # Either way, we need to delete it.
        logger.info("Deleting %s with size %d." % (leaf, size))
        self._synsets.pop(leaf)
        self.__synset_sizes.pop(leaf)

      else:
        # In this case, just take the parent automatically, since we're done
        # merging up to that leaf. The parent might be too small, in which case
        # we'll merge it with its parent.
        if leaf in mappings:
          leaves.add(mappings[leaf])

    while True:
      sizes_to_remove = []
      for synset, size in self.__synset_sizes.iteritems():
        merged_at_least_one = False
        if size < minimum_synset_size:
          # Try merging it.
          merge_with_superset(synset)

          # Delete it since it's too small.
          logger.info("Deleting %s with size %d." % (synset, size))
          self._synsets.pop(synset)
          # We can't pop items from the dictionary while we're iterating.
          sizes_to_remove.append(synset)

          merged_at_least_one = True

      for synset in sizes_to_remove:
        self.__synset_sizes.pop(synset)

      if not merged_at_least_one:
        # We're done here.
        break

    if len(self._synsets) <= target_categories:
      # We're done.
      logger.info("Ended up with %d synsets." % (len(self._synsets)))
      return

    # Sort our synsets from smallest to largest.
    sorted_sizes = sorted(self.__synset_sizes.items(),
                          key=operator.itemgetter(1))
    # We do a second pass in order to reach our target number of synsets.
    synset_index = 0
    needs_deleting = False
    while len(self._synsets) > target_categories:
      # Get the smallest one.
      synset, size = sorted_sizes[synset_index]
      synset_index += 1

      # Try merging it.
      if merge_with_superset(synset):
        # Delete it.
        logger.info("Deleting %s with size %d." % (synset, size))
        self._synsets.pop(synset)
        self.__synset_sizes.pop(synset)

      if synset_index >= len(sorted_sizes):
        # We got all the way through without reaching our target.
        needs_deleting = True
        break

    if needs_deleting:
      # At this point, we just delete the smallest ones until we have cut it
      # down to the proper size.
      logger.info("Deleting extra synsets.")
      # They changed, so we have to re-sort now.
      sorted_sizes = sorted(self.__synset_sizes.items(),
                            key=operator.itemgetter(1))
      synset_index = 0
      while len(self._synsets) > target_categories:
        synset, size = sorted_sizes[synset_index]
        synset_index += 1

        logger.info("Deleting %s with size %d." % (synset, size))
        self._synsets.pop(synset)
        self.__synset_sizes.pop(synset)

  def __save_synset(self, synset, data):
    """ Saves a synset to a file.
    Args:
      synset: The name of the synset.
      data: The data that will be saved in the file. """
    if not self._synset_location:
      # We're not saving synsets.
      return

    logger.info("Saving synset: %s" % (synset))

    synset_path = os.path.join(self._synset_location, "%s.json" % (synset))
    synset_file = open(synset_path, "w")
    json.dump(data, synset_file)
    synset_file.close()

  def _load_synsets(self, load_synsets=None):
    """ Loads synsets from a file.
    Args:
      load_synsets: If specified, this should be a set of synset names. It will
                    then only load synsets that are in this set.
    Returns:
      A set of the synsets that were loaded successfully. """
    loaded = set([])
    loaded_text = []
    logger.info("Loading synsets...")

    for path in os.listdir(self._synset_location):
      if path.endswith(".json"):
        # Load synset from file.
        synset_name = path[:-5]
        if load_synsets and synset_name not in load_synsets:
          # We don't have to load this one.
          continue

        logger.debug("Loading synset %s." % (synset_name))
        full_path = os.path.join(self._synset_location, path)

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
      file_path = os.path.join(self._synset_location, file_name)
      file_object = file(file_path)
      synset_data = json.load(file_object)
      file_object.close()

      # Remove the correct entries.
      try:
        synset_data.remove([wnid.decode("utf8"), url.decode("utf8")])
      except ValueError:
        logger.warning("%s, %s not in synset data?" % (wnid.decode("utf8"),
                                                       url.decode("utf8")))
        continue

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

  def save_datasets(self, file_prefix):
    """ Saves the datasets to the disk.
    Args:
      file_prefix: The first part of the filename for each dataset.
      "_training.pkl" and "_testing.pkl" will be appended to it for each
      individual datatset. """
    self._train_set.save_images(file_prefix + "_training.pkl")
    self._test_set.save_images(file_prefix + "_testing.pkl")

  def load_datasets(self, file_prefix):
    """ Loads the datasets from the disk.
    Args:
      file_prefix: The first part of the filename for each dataset.
      "_training.pkl" and "_testing.pkl" will be appended to it for each
      individual datatset. """
    self._train_set.load_images(file_prefix + "_training.pkl")
    self._test_set.load_images(file_prefix + "_testing.pkl")


class SynsetListImageGetter(ImageGetter):
  """ Works like an ImageGetter, but only loads images from a predefined set of
  synsets. """

  def __init__(self, synsets, *args, **kwargs):
    """
    Args:
      synsets: The list of synsets to load images from. """
    logger.debug("Using synsets: %s" % (synsets))

    self.__synset_names = set(synsets)

    super(SynsetListImageGetter, self).__init__(*args, **kwargs)

  def _populate_synsets(self):
    """ See documentation for superclass method. """
    if not os.path.exists(self._synset_location):
      os.mkdir(self._synset_location)

    loaded = self._load_synsets(load_synsets=self.__synset_names)
    self._get_image_list(loaded)

  def _get_image_list(self, loaded):
    """ See documentation for superclass method. """
    # Instead of downloading the complete list, we're just going to inject our
    # synsets here.
    synsets = self.__synset_names - loaded
    self._get_synset_urls(synsets)


class SynsetFileImageGetter(SynsetListImageGetter):
  """ Works like an ImageGetter, but only loads images from a predefined list of
  synsets specified in a file. """

  def __init__(self, synset_path, *args, **kwargs):
    """
    Args:
      synset_path: The path to the file containing a list of synsets, with one
      synset on each line. """
    # Parse the file.
    synset_file = file(synset_path)
    synsets = [synset.rstrip("\n") for synset in synset_file]
    synset_file.close()

    super(SynsetFileImageGetter, self).__init__(synsets, *args, **kwargs)


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

  def _pick_random_image(self):
    """ Picks a random image from our database.
    Returns:
      wnid and url, and key of the image in the images map. """
    # Pick a random image.
    wnid, url = self.__images.get_random()

    if wnid in self.__already_picked:
      # This is a duplicate, pick another.
      return self._pick_random_image()
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
        loaded_from_cache += self._load_random_image()

      downloaded = self._download_manager.update()
      successfully_downloaded += downloaded + loaded_from_cache

      time.sleep(0.2)

    logger.debug("Finished downloading.")
    self.__num_downloading -= self._batch_size
    return self._mem_buffer.get_batch(), to_remove

  def _load_random_image(self):
    """ Loads a random image from either the cache or the internet. If loading
    from the internet, it adds it to the download manager, otherwise, it adds
    them to the memory buffer.
    Returns:
      How many images it loaded from the cache successfully. """
    wnid, url = self._pick_random_image()
    synset, number = wnid.split("_")

    image = self._get_cached_image(synset, number)
    if image is None:
      # We have to download the image instead.
      if not self._download_manager.download_new(synset, number, url):
        # We're already downloading that image.
        logger.info("Already downloading %s. Picking new one..." % (wnid))
        return self._load_random_image()
      return 0

    # Cache hit.
    self._mem_buffer.add(image, number, synset)
    return 1

  def _get_cached_image(self, synset, image_number):
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
    image_file = file(filename, "rb")
    logger.info("Loading dataset from file: %s" % (filename))
    self.__images = pickle.load(image_file)

  def prune_images(self, synsets):
    """ Remove any images from the dataset that are not specified in the input.
    Args:
      synsets: The synset data structure containing all the acceptable images.
      If something is not in here, it will be removed.
    Returns:
      True if images were removed, False if none were. """
    logger.info("Pruning dataset...")

    # Put everything into a valid set.
    valid = set()
    for synset in synsets.keys():
      for wnid, url in synsets[synset]:
        valid.add((wnid, url))

    # Remove anything that's not in it.
    removed_image = False
    for wnid, url in self.__images:
      if (wnid, url) not in valid:
        self.__images.remove((wnid, url))
        removed_image = True

    return removed_image


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

  def _load_random_image(self):
    """ See superclass documentation. This override is necessary to deal with
    multiple patches. """
    wnid, url = self._pick_random_image()
    synset, number = wnid.split("_")

    patches = self._get_cached_image(synset, number)
    if patches is None:
      # We have to download the image instead.
      self._download_manager.download_new(synset, number, url)
      return 0

    # Cache hit.
    self._mem_buffer.add_patches(patches, number, synset)
    return 1

  def _get_cached_image(self, synset, image_number):
    """ See superclass documentation. This override is necessary to deal with
    multiple patches. """
    # First check in the cache for the image.
    cached_image = self._cache.get(synset, image_number)
    if cached_image is not None:
      # We had a cached copy, so we're done.
      logger.debug("Found image in cache: %s_%s" % (synset, image_number))

      # Extract patches.
      return data_augmentation.extract_patches(cached_image)

    return None

