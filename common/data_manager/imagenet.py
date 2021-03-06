import json
import logging
import operator
import os
import random
import urllib2

import dataset
import image_getter
import utils


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

    # We use a custom encoding for image IDs, so we have to convert.
    synset, name = wnid.split("_")
    img_id = utils.make_img_id(synset, name)

    mappings.append([img_id, url])
    if get_utf:
      utf_mappings.append([img_id, unicode_url])

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


class ImagenetGetter(image_getter.ImageGetter):
  """ Image getter specialization specifically for loading data from Imagenet.
  """

  def __init__(self, synset_location, *args, **kwargs):
    """
    Args:
      synset_location: Where to save synsets. Will be created if it doesn't
                       exist.
      For all other arguments, see the superclass version of this method.
      NOTE: If load_datasets_from is not specified, it will create new training
      and testing sets. """
    # Initialize the synsets before initializing the datasets.
    self._synset_location = synset_location

    self._synsets = {}
    self._populate_synsets()

    # Calculate and store sizes for each synset.
    self.__synset_sizes = {}
    self.__total_images = 0
    for synset, urls in self._synsets.iteritems():
      self.__synset_sizes[synset] = len(urls)
      self.__total_images += len(urls)

    super(ImagenetGetter, self).__init__(*args, **kwargs)

  def _init_datasets(self):
    """ Initializes the training and testing datasets. """
    # Make internal datasets for training and testing.
    if self._load_datasets_from:
      # Initialize empty datasets.
      self._make_new_datasets(set(), set())
      # Use the saved datasets instead of making new ones.
      self.load_datasets()

      # We might have deleted stuff since we last saved it, so we want to do
      # some pruning.
      if self._test_set.prune_images(self._synsets):
        logger.info("Updating test database file...")
        self._test_set.save_images()
      if self._train_set.prune_images(self._synsets):
        logger.info("Updating train database file...")
        self._train_set.save_images()

    else:
      train, test = self.__split_train_test_images(self._test_percentage)
      self._make_new_datasets(train, test)

      if self._load_datasets_from:
        # If we specified a path to load datasets from, make them here for next
        # time.
        logger.warning("Datasets not found. Making new ones and saving in %s." \
                       % (self._load_datasets_from))
        self.save_datasets()

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
      synset, _ = utils.split_img_id(wnid)

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


class SynsetListImagenetGetter(ImagenetGetter):
  """ Works like an ImageGetter, but only loads images from a predefined set of
  synsets. """

  def __init__(self, synsets, *args, **kwargs):
    """
    Args:
      synsets: The list of synsets to load images from. """
    logger.debug("Using synsets: %s" % (synsets))

    self.__synset_names = set(synsets)

    super(SynsetListImagenetGetter, self).__init__(*args, **kwargs)

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


class SynsetFileImagenetGetter(SynsetListImagenetGetter):
  """ Works like an ImagenetGetter, but only loads images from a predefined list of
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

    super(SynsetFileImagenetGetter, self).__init__(synsets, *args, **kwargs)


class FilteredImagenetGetter(ImagenetGetter):
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

    super(FilteredImagenetGetter, self).__init__(None, cache_location, batch_size, **kwargs)

  def _populate_synsets(self):
    """ Populates the synsets dict. """
    # Load data from the url file.
    urls = file(self.__url_file).read()
    images = _parse_url_file(urls, separator="\t")

    # Separate it by synset and load it into the dictionary.
    for wnid, url in images:
      synset, _ = utils.split_img_id(wnid)

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
