#!/usr/bin/python

import logging
import os

import cv2

import image_getter

def main():
  # Configure root logger.
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  file_handler = logging.FileHandler("/home/theano/server.log")
  file_handler.setLevel(logging.DEBUG)
  stream_handler = logging.StreamHandler()
  stream_handler.setLevel(logging.INFO)
  formatter = logging.Formatter("%(name)s@%(asctime)s: " +
      "[%(levelname)s] %(message)s")
  file_handler.setFormatter(formatter)
  stream_handler.setFormatter(formatter)
  root.addHandler(file_handler)
  root.addHandler(stream_handler)

  root.info("Starting...")

  training_data_path = "/home/theano/training_data/"
  synset_path = os.path.join(training_data_path, "synsets")
  cache_path = os.path.join(training_data_path, "cache")

  getter = image_getter.ImageGetter(synset_path, cache_path, 10)
  batch = getter.get_random_train_batch()
  for image in batch:
    cv2.imshow("test", image)
    cv2.waitKey(0)

main()
