""" Deals with image acquiring and manipulation. """


import httplib
import logging
import os
import re
import socket
import ssl
import urllib2
import urlparse
import time

import cv2

import numpy as np


logger = logging.getLogger(__name__)

# The error_images directory should be included in the same location that this
# file is.
_this_dir = os.path.dirname(os.path.realpath(__file__))
BAD_IMAGES_DIR = os.path.join(_this_dir, "error_images")
# How close an image has to be to a known bad image to throw it out.
ERROR_IMAGE_THRESH = 1.0
# Minimum number of bytes we need to consistently read every second before we
# give up.
MIN_DOWNLOAD_RATE = 512


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

def download_image(url, shape, keep_color=False):
  """ Downloads the image from the specified url.
  Args:
    url: The URL to download from.
    shape: The x and y shape we want the final image to be.
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
          socket.timeout, socket.error, ssl.CertificateError,
          httplib.HTTPException, httplib.IncompleteRead) as e:
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

  raw_data = ""
  slow_cycles = 0
  while True:
    start_time = time.time()
    try:
      new_data = response.read(512)
    except (socket.timeout, ssl.SSLError, socket.error) as e:
      logger.warning("Image download failed with '%s'." % (e))
      return None
    elapsed = time.time() - start_time

    if not new_data:
      # We're done reading the response.
      break

    raw_data += new_data

    if 512.0 / elapsed < MIN_DOWNLOAD_RATE:
      logger.debug("Downloading image too slowly.")
      slow_cycles += 1

      if slow_cycles >= 3:
        logger.warning("Aborting download due to slowness: %s" % (url))
        return None
    else:
      slow_cycles = 0

  image = np.asarray(bytearray(raw_data), dtype="uint8")
  if keep_color:
    flags = cv2.IMREAD_COLOR
  else:
    flags = cv2.IMREAD_GRAYSCALE
  image = cv2.imdecode(image, flags)
  if image is None:
    return image

  # Reshape the image.
  image = reshape_image(image, shape)

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

def reshape_image(image, shape, offset=(0, 0)):
  """ Reshapes a stored image so that it is a consistent shape and size.
  Args:
    image: The image to reshape.
    shape: The shape we want the image to be.
    offset: An optional offset. This can be used to direct it not to crop to the
            center of the image. In the tuple, the horizontal offset comes
            before the vertical one.
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
  target_width, target_height = shape

  # Find the largest we can make the initial crop.
  multiplier = 1
  if width > target_width:
    multiplier = width / target_width
  elif height > target_height:
    multiplier = height / target_height
  target_width *= multiplier
  target_height *= multiplier

  crop_width = target_width
  crop_height = target_height
  # Our goal here is to keep the same aspect ratio as the original.
  if width <= target_width:
    # We need to reduce the width for our initial cropping.
    crop_width = width
    crop_height = target_height * (float(crop_width) / target_width)
  if height <= target_height:
    # We need to reduce the height for our initial cropping.
    crop_height = height
    crop_width = target_width * (float(crop_height) / target_height)

  crop_width = int(crop_width)
  crop_height = int(crop_height)

  logger.debug("Cropping to size: (%d, %d)" % (crop_width, crop_height))

  # Crop the image.
  crop_left = (width - crop_width) / 2
  crop_top = (height - crop_height) / 2

  # Account for the crop offset.
  offset_left, offset_top = offset
  crop_left += offset_left
  crop_top += offset_top
  # Bound it in the image.
  crop_left = max(0, crop_left)
  crop_left = min(width - 1, crop_left)
  crop_top = max(0, crop_top)
  crop_top = min(height - 1, crop_top)

  image = image[crop_top:(crop_height + crop_top),
                crop_left:(crop_width + crop_left)]

  # Set a proper size, which should just be directly scaling up or down.
  image = cv2.resize(image, shape)

  return image


socket.setdefaulttimeout(10)

# Pre-load error images.
if not os.path.exists(BAD_IMAGES_DIR):
  raise RuntimeError("Could not find bad images directory '%s'." % \
                     (BAD_IMAGES_DIR))
else:
  _error_images = _load_error_images()
