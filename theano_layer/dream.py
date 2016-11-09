#!/usr/bin/python


""" Neural network dreaming code. This is adapted mainly from here:
https://github.com/google/deepdream/blob/master/dream.ipynb """


import random
import sys

import cv2

import numpy as np
import scipy.ndimage as nd

import theano

from alexnet import AlexNet


def _show_thumbnails(images, wait=False):
  """ Show thumbnails of a set of images, and wait for user input.
  Args:
    images: The images to show.
    wait: Whether to wait for user input to continue. """
  film_strip = np.concatenate([np.transpose(i, (1, 2, 0)) \
                               for i in images[0:8]], axis=1)
  cv2.imshow("dream", film_strip.astype("uint8"))
  if wait:
    cv2.waitKey(0)
  else:
    cv2.waitKey(1)

def _make_step(network, image, step_size=1.5, jitter=32, clip=True):
  """ Perform gradient ascent to maximize the L2 norms of the activations in a
  layer.
  Args:
    network: The network to use.
    image: The input image to use.
    step_size: The size of the gradient ascent step.
    jitter: The amount of jitter to add to the input.
    clip: If we want to clip the output data so that it stays in range. """
  # Add jitter to the input image.
  ox, oy = np.random.randint(-jitter, jitter + 1, 2)
  image = np.roll(np.roll(image, ox, -1), oy, -2)

  # Write the image into the buffer.
  test_x = network.get_test_x()
  test_x.set_value(image)

  # Specify the objective and calculate input gradients.
  grads = network.l2_norm_backwards(0)[0]

  # Apply normalized ascent step to the input image.
  image += step_size / np.abs(grads).mean() * grads

  # Unjitter image.
  image = np.roll(np.roll(image, -ox, -1), -oy, -2)

  if clip:
    image = np.clip(image, -127, 127)

  return image

def make_dream(network, start, iter_n=10, octave_n=4, octave_scale=1.4,
               clip=True, **step_params):
  """ Creates a new dream.
  Args:
    network: The network to dream with.
    start: The starting image.
    iter_n: The number of iterations to perform.
    octave_n: The number of octaves.
    octave_scale: The scale of the octaves.
  Returns:
    The created dream. """
  octaves = [start.copy()]
  for i in xrange(octave_n - 1):
    octaves.append(nd.zoom(octaves[-1],
                   (1, 1, 1.0 / octave_scale,
                    1.0 / octave_scale),
                    order=1))

  input_h, input_w = start.shape[-2:]

  # Allocate image for network-produced details.
  details = np.zeros_like(octaves[-1])
  for octave, octave_base in enumerate(octaves[::-1]):
    height, width = octave_base.shape[-2:]
    if octave > 0:
      # Upscale details from previous octave.
      old_height, old_width = details.shape[-2:]
      details = nd.zoom(details, (1, 1, 1.0 * height / old_height,
                                        1.0 * width / old_width), order=1)

    start = octave_base + details
    _show_thumbnails(start + 127)
    for i in xrange(iter_n):
      # Run the gradient ascent step.
      start = _make_step(network, start, clip=clip, **step_params)
      # Visualize.
      _show_thumbnails(start + 127)

    # Extract details produced on the current octave.
    details = start - octave_base

  return start

def main():
  if len(sys.argv) != 3:
    print "Usage: %s network_file input_image" % (sys.argv[0])
    return

  input_image = sys.argv[2]

  # Create random input image.
  #random_image = np.random.uniform(-127, 127, (128, 3, 224, 224))
  #random_image = random_image.astype(theano.config.floatX)
  # Load input image.
  raw_image = cv2.imread(input_image)
  raw_image = raw_image.transpose(2, 0, 1)
  image = np.zeros((1,) + raw_image.shape)
  image[0] = raw_image
  # Load test set with image.
  image -= 127
  image = image.astype("float32")
  test = theano.shared(image)

  # Load the network.
  network_file = sys.argv[1]
  print "Theano: Loading network from file..."
  network = AlexNet.load(network_file, None, (test, None), 1)
  print "Done."

  # Do the dream-making.
  s = 0.05
  height, width = image.shape[:-2]
  i = 0
  while True:
    image = make_dream(network, image)
    # Save the image.
    cv2.imwrite("dreams/dream%d.jpg" % (i), image[0].transpose(1, 2, 0) + 127)
    i += 1

    image = nd.affine_transform(image, [1, 1, 1 - s, 1 - s],
                                       [0, 0, height * s / 2, width * s / 2],
                                order=1)
    print "Next scale factor."

if __name__ == "__main__":
  main()
