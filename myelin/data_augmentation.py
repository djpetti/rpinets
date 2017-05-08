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
    The five extracted patches. """
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

def pca(image, stddev=0.1):
  """ Performs Principle Component Analysis on input image and adjusts it
  randomly, simluating different lighting intensities.
  Args:
    image: Input image matrix.
    stddev: The standard deviation of the random scale variable.
  Return:
    Adjusted image.
  """
  # Scale between 0 and 1.
  image = image.astype("float32") / 255.0

  # Reshape image.
  num_pixels = image.shape[0] * image.shape[1]
  reshaped_image = np.reshape(image, (num_pixels, 3))
  # Find the covariance.
  cov = np.cov(reshaped_image, rowvar=False)
  print cov
  # Eigenvalues and vectors.
  eigvals, eigvecs = np.linalg.eigh(cov)

  # Pick random gaussian values.
  alpha = np.random.normal(0, stddev, size=(3,))

  scaled = eigvals * alpha
  delta = np.dot(eigvecs, scaled.T)
  transformed = image + delta

  # Convert back to ints.
  transformed *= 255.0
  transformed = np.clip(transformed, 0, 255)
  return transformed.astype("uint8")

def jitter(image, stddev=0.1):
  """ Add some random noise to the image.
  Args:
    image: Input image matrix.
    stddev: The standard deviation of the noise. """
  # Generate noise.
  noise = np.random.normal(1.0, stddev, size=image.shape)
  noisy = image.astype("float32") * noise

  # Convert back to ints.
  noisy = np.clip(noisy, 0, 255)
  return noisy.astype("uint8")
