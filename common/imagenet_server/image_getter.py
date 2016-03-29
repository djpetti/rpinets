import json
import logging
import os
import random
import urllib2

import cache
import images


logger = logging.getLogger(__name__)


class ImageGetter(object):
  """ Gets random sets of images for use in training and testing. """

  def __init__(self):
    self.__cache = cache.DiskCache("image_cache", 1586113)

    if not self.__load_synsets():
      self.__download_image_list()

    # Calculate and store sizes for each synset.
    self.__synset_sizes = {}
    for synset, urls in self.__synsets.iteritems():
      self.__synset_sizes[synset] = len(urls)

  def __download_image_list(self):
    """ Downloads a comprehensive list of all images available. """
    logger.info("Downloading list of synsets...")
    url = "http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list"
    response = urllib2.urlopen(url, timeout=1000)
    synsets = response.read().split("\n")[:2]
    logger.debug("Got synsets: %s", synsets)

    # Get the urls for each synset.
    self.__synsets = {}
    base_url = \
        "http://www.image-net.org/api/text/" \
        "imagenet.synset.geturls.getmapping?wnid=%s"
    for synset in synsets:
      logger.info("Downloading urls for synset: %s", synset)
      response = urllib2.urlopen(base_url % (synset), timeout=1000)
      lines = response.read().split("\n")[:-1]
      # Split ids and urls.
      mappings = []
      for line in lines:
        if not line.startswith(synset):
          # Bad line.
          continue
        wnid, url = line.split()
        mappings.append([wnid, url])
      self.__synsets[synset] = mappings

    # Save them for later.
    self.__save_synsets()

  def __save_synsets(self):
    """ Saves synsets to a file. """
    logging.info("Saving synsets.")

    synset_file = open("synsets.json", "w")
    json.dump(self.__synsets, synset_file)
    synset_file.close()

  def __load_synsets(self):
    """ Loads synsets from a file.
    Returns:
      True if the data was loaded successfully, False otherwise. """
    if not os.path.exists("synsets.json"):
      return False

    logging.info("Loading synsets.")
    synset_file = open("synsets.json")
    self.__synsets = json.load(synset_file)
    synset_file.close()

    return True

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
        self.__save_synsets()
        continue

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
