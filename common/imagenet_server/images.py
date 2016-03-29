""" Deals with image acquiring and manipulation. """


import logging
import urllib2

import cv2

import numpy as np


logger = logging.getLogger(__name__)


def download_image(url):
  """ Downloads the image from the specified url.
  Args:
    url: The URL to download from.
  Returns:
    The image data that was downloaded.
  """
  logger.info("Downloading new image: %s", url)

  try:
    response = urllib2.urlopen(url, timeout=10)
  except (urllib2.HTTPError, urllib2.URLError) as e:
    # Generally, this is because the image was not found.
    logger.warning("Image download failed with '%s'." % (e))
    return None

  if "photo_unavailable" in response.geturl():
    # Flickr has this wonderful failure mode where it just redirects to this
    # picture instead of throwing a 404 error.
    logger.warning("Got Flickr 'photo unavailable' error.")
    return None

  raw_data = response.read()

  image = np.asarray(bytearray(raw_data), dtype="uint8")
  return cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

def download_words(wnid):
  """ Downloads the words associated with a synset.
  Args:
    wnid: The wnid of the synset.
  Returns:
    The word data that was downloaded. """
  base_url = \
      "http://www.image-net.org/api/text/wordnet.synset.getwords?wnid=%s"
  response = urllib2.urlopen(base_url % (wnid))
  words = response.read().split("\n")[:-1]
  logger.info("Got words for synset: %s", words)
  return words

def reshape_image(image):
  """ Reshapes a stored image so that it is a consistent shape and size.
  Args:
    image: The image to reshape.
  Returns:
    The reshaped image.
  """
  # Crop the image to just the center square.
  height, width = image.shape
  logger.debug("Original image shape: (%d, %d)" % (width, height))
  if width != height:
    if width > height:
      # Landscape
      length = height
      crop_left = (width - height) / 2
      crop_top = 0
    elif height > width:
      # Portrait.
      length = width
      crop_top = (height - width) / 2
      crop_left = 0
    image = image[crop_top:(length + crop_top), crop_left:(length + crop_left)]

  # Set a proper size. At this point, we'll do 256x256, which should be enough
  # resolution for simple classification.
  image = cv2.resize(image, (256, 256))

  return image
