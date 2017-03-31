#!/usr/bin/python


""" Convert an existing dataset to the format used by data_manager. """


import argparse
import os
import random
import sys

import cv2

import cache
import dataset
import images
import utils


def _process_label(label, path, image_list):
  """ Processes all the images in a single label.
  Args:
    label: The name of the label.
    path: The base path where all the label directories are.
    image_list: The list containing tuples of the lable, name, and path
    for all the images loaded so far. This will be added to. """
  print "Processing label: %s" % (label)

  label_path = os.path.join(path, label)

  for image in os.listdir(label_path):
    # Load the image from the disk.
    image_path = os.path.join(label_path, image)
    image_list.append((label, image, image_path))

def _build_dataset(dataset_images, disk_cache, size, offset):
  """ Builds a new dataset, and adds images to the disk cache.
  Args:
    dataset_images: The images to include in the dataset. These should be tuples
                    of the label, image name, and full image path.
    disk_cache: The DiskCache to add loaded images to.
    size: The x and y size of each image in the dataset.
    offset: The x and y offset to use when resizing each image.
  Returns:
    dataset: The dataset that was built from the data. """
  # Shuffle the images, and load them into the cache randomly. This is so that
  # contiguous loading from the cache actually works.
  print "Building image cache..."
  random.shuffle(dataset_images)

  print "Processing %d images..." % (len(dataset_images))

  # Load them into the cache.
  processed = 0
  last_percentage = 0
  for label, name, path in dataset_images:
    image_data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image_data is None:
      print "WARNING: Failed to load image: '%s'." % (path)
      continue

    # Reshape the image.
    image_data = images.reshape_image(image_data, size, offset=offset)
    # Add to the cache.
    disk_cache.add(image_data, name, label)

    # Print percentage done.
    percentage = int(float(processed) / len(dataset_images) * 100)
    if percentage != last_percentage:
      print "(%d percent done.)" % (percentage)
      last_percentage = percentage

    processed += 1

  # Create and save a dataset. (The batch size and image size arguments don't
  # really matter here.)
  dataset_entries = []
  for label, name, _ in dataset_images:
    img_id = utils.make_img_id(label, name)
    # The None in the tuple is because there are no URLs corresponding to these
    # images.
    dataset_entries.append((img_id, None))
  data = dataset.Dataset(dataset_entries, disk_cache, 0, (0, 0, 0))
  return data

def collect_dataset_images(path):
  """ Collects two lists of the images in the training and testing sets.
  Args:
    path: The location of the dataset.
  Returns: A list of the images in the dataset. """
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
    _process_label(label, path, dataset_images)

  return dataset_images

def divide_sets(path, test_fraction=0.1):
  """ Divides a set of images into a training set and testing set.
  Args:
    path: The path to the image directory. This should contain folders for
              each label.
    test_fraction: What fraction of the images should be used for a test set.
  Returns: The list of training images, and the list of testing images.
  """
  # First, we'll load everything, and then split it later.
  all_images = collect_dataset_images(path)

  print "Generating new train/test split."

  # Split everything.
  train_images = []
  test_images = []
  for image in all_images:
    if random.random() <= test_fraction:
      test_images.append(image)
    else:
      train_images.append(image)

  return (train_images, test_images)

def convert_dataset(location, size, output, do_split=False, offset=(0, 0)):
  """ Converts a dataset to a format that's usable by data_manager.
  Args:
    location: The location of the dataset to convert.
    size: The x and y sizes of each image in the dataset.
    output: The location to write output files to.
    do_split: Whether to generate a new train/test split or use an existing one.
    offset: Optional offset to use for width and height when resizing the image.
  """
  # The conversion works by reading all the images and adding them to a
  # DiskCache.
  disk_cache = cache.DiskCache(output)

  if not do_split:
    # Individual categories are contained in the train and test directories.
    train_path = os.path.join(location, "train")
    test_path = os.path.join(location, "test")

    train_images = collect_dataset_images(train_path)
    test_images = collect_dataset_images(test_path)

  else:
    # We'll have to generate a train/test split.
    train_images, test_images = divide_sets(location)

  train_set = _build_dataset(train_images, disk_cache, size, offset)
  test_set = _build_dataset(test_images, disk_cache, size, offset)

  # Save the datasets to the disk.
  print "Saving dataset_train.pkl..."
  train_path = os.path.join(output, "dataset_training.pkl")
  train_set.save_images(train_path)
  print "Saving dataset_test.pkl..."
  test_path = os.path.join(output, "dataset_testing.pkl")
  test_set.save_images(test_path)

  print "Done."

def main():
  # Parse user arguments.
  parser = argparse.ArgumentParser( \
      description="Convert an existing dataset to a usable form.")
  parser.add_argument("width", type=int, help="The width of converted images.")
  parser.add_argument("height", type=int, help="The height of converted images.")
  parser.add_argument("dataset", help="The location of the dataset to convert.")
  parser.add_argument("-o", "--output", default=".",
                      help="Location to write output to.")
  parser.add_argument("-w", "--offset_width", default=0, type=int,
                      help="Width offset for each image.")
  parser.add_argument("-H", "--offset_height", default=0, type=int,
                      help="Height offset for each image.")
  parser.add_argument("-s", "--split_train_test", action="store_true",
                      help="Whether to generate a new train/test split.")
  args = parser.parse_args()

  convert_dataset(args.dataset, (args.width, args.height), args.output,
                  do_split=args.split_train_test, offset=(args.offset_width,
                                                          args.offset_height))


if __name__ == "__main__":
  main()
