import random

import numpy as np


def extract_patches(image, patch_shape, flip=True):
  """ Extract patches from the image. It extracts ten such patches:
  Top left, top right, bottom left, bottom right, and center, plus horizontal
  reflections of them all.
  Args:
    image: The input image to extract patches from.
    patch_shape: A two-element tuple containing the size of each patch.
    flip: If this is False, it won't extract horizontal reflections.
  Returns:
    The five or ten extracted patches. """
  # Get the initial image shape.
  width, height, _ = image.shape
  new_width, new_height = patch_shape

  top_left = image[0:new_width, 0:new_height]
  top_right = image[width - new_width:width, 0:new_height]
  bottom_left = image[0:new_width, height - new_height:height]
  bottom_right = image[width - new_width:width, height - new_height:height]

  distx_from_edge = (width - new_width) / 2
  disty_from_edge = (height - new_height) / 2
  center = image[distx_from_edge:width - distx_from_edge,
                 disty_from_edge:height - disty_from_edge]

  ret = [top_left, top_right, bottom_left, bottom_right, center]

  if flip:
    # Flip everything as well.
    top_left_flip = np.fliplr(top_left)
    top_right_flip = np.fliplr(top_right)
    bottom_left_flip = np.fliplr(bottom_left)
    bottom_right_flip = np.fliplr(bottom_right)
    center_flip = np.fliplr(center)

    ret.extend([top_left_flip, top_right_flip, bottom_left_flip,
                bottom_right_flip, center_flip])

  return ret

def extract_random_patch(image, patch_shape, flip=True):
  """ Extracts a single patch randomly from an image.
  Args:
    image: The input image to extract the patch from.
    patch_shape: A two-element tuple containing the size of each patch.
    flip: If this is False, it won't randomly flip the patch.
  Returns:
    The single extracted patch. """
  # Get the intial image shape.
  width, height, _ = image.shape
  new_width, new_height = patch_shape

  # Pick a random location for the top left corner of the patch.
  max_x = width - new_width
  max_y = height - new_height
  corner_x = random.randint(0, max_x)
  corner_y = random.randint(0, max_y)

  crop = image[corner_x:(corner_x + new_width),
               corner_y:(corner_y + new_height)]

  if (flip and random.randint(0, 1)):
    # We may randomly flip it horizontally.
    crop = np.fliplr(crop)

  return crop

def pca(image):
  """ Performs Principle Component Analysis on input image and adjusts it
  randomly, simulating different lighting intensities.
  Args:
    image: Input image matrix.
  Return:
    Adjusted image.
  """
  # Reshape image.
  reshaped_image = np.reshape(image, (224 * 224, 3))
  # Find the covariance.
  cov = np.cov(reshaped_image, rowvar=0)
  # Eigenvalues and vectors.
  eigvals, eigvecs = np.linalg.eigh(cov)

  # Pick random gaussian values.
  a = np.random.normal(0, 0.1, size=(3,))

  scaled = eigvals * a
  delta = np.dot(eigvecs, scaled.T)
  return np.add(delta, scaled)
