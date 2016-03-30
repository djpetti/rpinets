#!/usr/bin/python

import logging

import cv2

import image_getter

def main():
  # Configure root logger.
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  file_handler = logging.FileHandler("server.log")
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
  getter = image_getter.ImageGetter("synsets")
  batch = getter.get_random_batch(10)
  for image in batch:
    cv2.imshow("test", image)
    cv2.waitKey(0)

main()
