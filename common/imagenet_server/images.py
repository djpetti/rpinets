""" Deals with image acquiring and manipulation. """


import httplib
import logging
import os
import re
import socket
import ssl
import urllib2
import urlparse

import cv2

import numpy as np


logger = logging.getLogger(__name__)


BAD_IMAGES_DIR = "error_images"
# How close an image has to be to a known bad image to throw it out.
ERROR_IMAGE_THRESH = 1.0


def _load_error_images():
  """ Loads error images from the disk.
  Returns:
    The list of error images. """
  file_names = os.listdir(BAD_IMAGES_DIR)
  error_images = []
  for image_name in file_names:
    image_path = os.path.join(BAD_IMAGES_DIR, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
      raise RuntimeError("Loading bad image %s failed." % (image_path))
    error_images.append(image)

  return error_images

def _url_encode_non_ascii(b):
  """ Encodes non-ascii characters so they can be used in URLs """
  return re.sub('[\x80-\xFF]', lambda c: '%%%02x' % ord(c.group(0)), b)

def _iri_to_uri(iri):
  """ Convert an IRI to a URI. """
  parts= urlparse.urlparse(iri)
  encoded = []
  for parti, part in enumerate(parts):
    if parti == 1:
      encoded.append(part.encode("idna"))
    else:
      encoded.append(_url_encode_non_ascii(part.encode("utf-8")))
  return urlparse.urlunparse(encoded)

def _check_bad_image(image):
  """ Checks an image against a known set of bad images in order to decide
  whether this image is good or not.
  Returns: True if it thinks the image is bad, False otherwise. """
  # Check if it matches any of the error images.
  for error_image in _error_images:
    distance = abs(np.mean(image.astype("int8") - error_image.astype("int8")))
    if distance < ERROR_IMAGE_THRESH:
      return True

  return False

def download_image(url, keep_color=False):
  """ Downloads the image from the specified url.
  Args:
    url: The URL to download from.
    keep_color: If False, images will be saved in grayscale.
  Returns:
    The image data that was downloaded.
  """
  logger.info("Downloading new image: %s", url)
  try:
    url = _iri_to_uri(url)
  except UnicodeDecodeError as e:
    logger.warning("Error decoding URL: %s" % (e))
    return None

  try:
    response = urllib2.urlopen(url, timeout=10)
  except (urllib2.HTTPError, urllib2.URLError, httplib.BadStatusLine,
          socket.timeout, socket.error, ssl.CertificateError) as e:
    # Generally, this is because the image was not found.
    logger.warning("Image download failed with '%s'." % (e))
    return None

  if "photo_unavailable" in response.geturl():
    # Flickr has this wonderful failure mode where it just redirects to this
    # picture instead of throwing a 404 error. Actually, lots of websites do
    # this, but since Flickr is the most common one, it saves time to handle
    # that issue here.
    logger.warning("Got Flickr 'photo unavailable' error.")
    return None

  try:
    raw_data = response.read()
  except (socket.timeout, ssl.SSLError) as e:
    logger.warning("Image download failed with '%s'." % (e))
    return None

  image = np.asarray(bytearray(raw_data), dtype="uint8")
  if keep_color:
    flags = cv2.IMREAD_COLOR
  else:
    flags = cv2.IMREAD_GRAYSCALE
  image = cv2.imdecode(image, flags)
  if image is None:
    return image

  # Reshape the image.
  image = reshape_image(image)

  # Check for other bad images besides Flickr's.
  if _check_bad_image(image):
    logging.warning("Got bad image: %s." % (url))
    return None

  return image

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
  if len(image.shape) == 3:
    # It may have multiple color channels.
    height, width, _ = image.shape
  else:
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


socket.setdefaulttimeout(10)

# Pre-load error images.
if not os.path.exists(BAD_IMAGES_DIR):
  raise RuntimeError("Could not find bad images directory '%s'." % \
                     (BAD_IMAGES_DIR))
else:
  _error_images = _load_error_images()
