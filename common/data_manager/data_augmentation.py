import numpy as np


def extract_patches(image):
  """ Extracts 224x224 patches from the image. It extracts ten such patches:
  Top left, top right, bottom left, bottom right, and center, plus horizontal
  reflections of them all.
  Args:
    image: The input image to extract patches from.
  Returns:
    The five extracted patches. """
  top_left = image[0:224, 0:224]
  top_right = image[256 - 224:256, 0:224]
  bottom_left = image[0:224, 256 - 224:256]
  bottom_right = image[256 - 224:256, 256 - 224:256]

  distance_from_edge = (256 - 224) / 2
  center = image[distance_from_edge:256 - distance_from_edge,
                  distance_from_edge:256 - distance_from_edge]

  # Flip everything as well.
  top_left_flip = np.fliplr(top_left)
  top_right_flip = np.fliplr(top_right)
  bottom_left_flip = np.fliplr(bottom_left)
  bottom_right_flip = np.fliplr(bottom_right)
  center_flip = np.fliplr(center)

  return (top_left, top_left_flip, top_right, top_right_flip, bottom_left,
          bottom_left_flip, bottom_right, bottom_right_flip, center,
          center_flip)

def pca(image):
  """ Performs Principle Component Analysis on input image and adjusts it
  randomly, simluating different lighting intensities.
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
