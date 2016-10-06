#!/usr/bin/python


""" Convert an existing dataset to the format used by data_manager. """


import argparse
import os
import sys

import cv2

import cache
import dataset
import images
import utils


def _process_label(label, path, disk_cache, size):
  """ Processes all the images in a single label.
  Args:
    label: The name of the label.
    path: The base path where all the label directories are.
    disk_cache: The DiskCache to add image data to.
    size: The x and y size of each image in the dataset.
  Returns:
    A list of the images processed. Each item is a tuple with the ID of the
    image, formed by joining the label and the filename together with an
    underscore, and None, which will represent the URL attribute for insertion
    into a Dataset. """
  print "Processing label: %s" % (label)

  label_path = os.path.join(path, label)

  image_names = []
  for image in os.listdir(label_path):
    # Load the image from the disk.
    image_path = os.path.join(label_path, image)
    image_data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image_data is None:
      print "WARNING: Failed to load image: '%s'." % (image_path)
      continue

    # Reshape the image.
    image_data = images.reshape_image(image_data, size)

    disk_cache.add(image_data, image, label)

    image_id = utils.make_img_id(label, image)
    image_names.append((image_id, None))

  return image_names

def _build_dataset(path, disk_cache, size):
  """ Builds a new dataset, and adds images to the disk cache.
  Args:
    path: The path to the images, which should be separated by folder into
          categories.
    disk_cache: The DiskCache to add loaded images to.
    size: The x and y size of each image in the dataset.
  Returns:
    dataset: The dataset that was built from the data. """
  print "Building dataset from images at %s..." % (path)

  if not os.path.exists(path):
    print "ERROR: I tried my best, but I couldn't find '%s'." % (path)
    print "I'm really sorry, old chap."
    sys.exit(1)

  dataset_images = []
  for label in os.listdir(path):
    label_path = os.path.join(path, label)
    if not os.path.isdir(label_path):
      print "WARNING: Skipping extraneous item: '%s'" % (label_path)
      continue

    # Read the contents of the directory.
    dataset_images.extend(_process_label(label, path, disk_cache, size))

  # Create and save a dataset. (The batch size and image size arguments don't
  # really matter here.)
  data = dataset.Dataset(dataset_images, disk_cache, 0, (0, 0, 0))
  return data

def convert_dataset(location, size):
  """ Converts a dataset to a format that's usable by data_manager.
  Args:
    location: The location of the dataset to convert.
    size: The x and y sizes of each image in the dataset. """
  # The conversion works by reading all the images and adding them to a
  # DiskCache.
  disk_cache = cache.DiskCache(".")

  # Individual categories are contained in the train and test directories.
  train_path = os.path.join(location, "train")
  test_path = os.path.join(location, "test")
  train_set = _build_dataset(train_path, disk_cache, size)
  test_set = _build_dataset(test_path, disk_cache, size)

  # Save the datasets to the disk.
  print "Saving dataset_train.pkl..."
  train_set.save_images("dataset_training.pkl")
  print "Saving dataset_test.pkl..."
  test_set.save_images("dataset_testing.pkl")

  print "Done."

def main():
  # Parse user arguments.
  parser = argparse.ArgumentParser( \
      description="Convert an existing dataset to a usable form.")
  parser.add_argument("width", type=int, help="The width of converted images.")
  parser.add_argument("height", type=int, help="The height of converted images.")
  parser.add_argument("dataset", help="The location of the dataset to convert.")
  args = parser.parse_args()

  convert_dataset(args.dataset, (args.width, args.height))


if __name__ == "__main__":
  main()
