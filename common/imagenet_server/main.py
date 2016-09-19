#!/usr/bin/python

import logging

def _configure_logging():
  """ Configure logging handlers. """
  # Configure root logger.
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  file_handler = logging.FileHandler("test_image_getter.log")
  file_handler.setLevel(logging.DEBUG)
  stream_handler = logging.StreamHandler()
  stream_handler.setLevel(logging.INFO)
  formatter = logging.Formatter("%(name)s@%(asctime)s: " +
      "[%(levelname)s] %(message)s")
  file_handler.setFormatter(formatter)
  stream_handler.setFormatter(formatter)
  root.addHandler(file_handler)
  root.addHandler(stream_handler)

# Some modules need logging configured immediately to work.
_configure_logging()

import os

import cv2

import numpy as np

import image_getter

def main():
  logging.info("Starting...")

  getter = image_getter.SynsetFileImageGetter("use_synsets.txt", "synsets",
                                              "image_cache",
                                              10, preload_batches=2)

  for x in range(0, 3):
    batch = getter.get_random_train_batch()

    print batch[1]
    print len(batch[1])
    i = 0
    for image in batch[0]:
      print "Showing image: %d" % (i)
      i += 1
      cv2.imshow("test", np.transpose(image, (1, 2, 0)))
      cv2.waitKey(0)

main()
