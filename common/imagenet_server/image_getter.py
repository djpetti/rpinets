import json
import logging
import os
import random
import urllib2

import cv2

import cache
import images


logger = logging.getLogger(__name__)


class ImageGetter(object):
  """ Gets random sets of images for use in training and testing. """
  # Minimum synset size to use.
  MINIMUM_SYNSET_SIZE = 500

  def __init__(self, synset_location):
    """
    Args:
      synset_location: Where to save synsets. Will be created if it doesn't
      exist. """
    self.__cache = cache.DiskCache("image_cache", 50000000000)

    self.__synset_location = synset_location
    if not os.path.exists(self.__synset_location):
      os.mkdir(self.__synset_location)

    self.__synsets = {}
    loaded = self.__load_synsets()
    #self.__download_image_list(loaded)

    # Calculate and store sizes for each synset.
    self.__synset_sizes = {}
    for synset, urls in self.__synsets.iteritems():
      self.__synset_sizes[synset] = len(urls)

  def __download_image_list(self, loaded):
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
      lines = response.read().split("\n")[:-1]
      # Split ids and urls.
      mappings = []
      utf_mappings = []
      for line in lines:
        if not line.startswith(synset):
          # Bad line.
          continue

        wnid, url = line.split(" ", 1)
        try:
          unicode_url = url.decode("utf8")
        except UnicodeDecodeError:
          # We got a URL that's not valid unicode. (It does happen.)
          logger.warning("Skipping invalid URL: %s", url)
          continue

        mappings.append([wnid, url])
        utf_mappings.append([wnid, unicode_url])
      self.__synsets[synset] = mappings
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
    for path in os.listdir(self.__synset_location):
      if path.endswith(".json"):
        # Load synset from file.
        synset_name = path[:-5]
        logger.info("Loading synset %s." % (synset_name))
        full_path = os.path.join(self.__synset_location, path)

        synset_file = file(full_path)
        # Storing stuff in unicode takes up a massive amount of memory.
        self.__synsets[synset_name] = \
            [str(elem) for elem in json.load(synset_file)]
        synset_file.close()

        loaded.add(synset_name)

    return loaded

  def __pick_random_image(self):
    """ Picks a random image from our database.
    Returns:
      wnid and url of the image. """
    # Pick random synset.
    keys = self.__synsets.keys()
    synset_index = random.randint(0, len(keys) - 1)
    synset = keys[synset_index]

    # Pick random image from that synset.
    image_index = random.randint(0, self.__synset_sizes[synset] - 1)
    wnid, url = self.__synsets[synset][image_index]

    if wnid in self.__already_picked:
      # This is a duplicate, pick another.
      return self.__pick_random_image()
    self.__already_picked.add(wnid)

    return [wnid, url]

  def get_random_batch(self, size):
    """ Loads a random batch of images from the whole dataset.
    Args:
      size: The size of the batch to load.
    Returns:
      A list of the loaded images. """
    logger.info("Getting batch of size %d.", size)

    # Keeps track of images that were already used this batch.
    self.__already_picked = set([])

    # First, figure out which images we want.
    batch = []
    while len(batch) < size:
      wnid, url = self.__pick_random_image()

      image = self.__get_image_from_url(wnid, url)
      if image == None:
        # Remove the image.
        logger.info("Removing bad image %s." % (wnid))
        synset, _ = wnid.split("_")
        self.__synsets[synset].remove([wnid, url])
        self.__synset_sizes[synset] -= 1
        # Update the json file.
        self.__save_synset(synset)
        continue

      cv2.imshow("loading", image)
      # Force it to update the window.
      cv2.waitKey(1)
      batch.append(image)

    return batch

  def __get_image_from_url(self, wnid, url):
    """ Given a URL, it gets the corresponding image from the best location
    possible.
    Args:
      wnid: The wnid (with image number) of the image in question.
      url: The URL of the image.
    Returns:
      The actual image data, or None if the download failed. """
    # First check in the cache for the image.
    synset, image_number = wnid.split("_")
    cached_image = self.__cache.get(synset, image_number)
    if cached_image != None:
      # We had a cached copy, so we're done.
      logger.debug("Found image in cache: %s", wnid)
      return cached_image

    # Otherwise, we have to download it.
    downloaded_image = images.download_image(url)
    if downloaded_image == None:
      # Download failed.
      return None
    # Reshape it too so it's suitable for use.
    downloaded_image = images.reshape_image(downloaded_image)
    # Add it to the cache.
    self.__cache.add(downloaded_image, image_number, synset)

    return downloaded_image

  def get_synsets(self):
    return self.__synsets
