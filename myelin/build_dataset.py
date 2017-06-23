#!/usr/bin/python


""" Convert an existing dataset to the format used by data_manager. """


import argparse
import cPickle as pickle
import os
import random
import sys
import time

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
    image_list: The list containing tuples of the label, name, and path
    for all the images loaded so far. This will be added to. """
  print "Processing label: %s" % (label)

  label_path = os.path.join(path, label)

  for image in os.listdir(label_path):
    # Add the image.
    image_path = os.path.join(label_path, image)
    image_list.append((label, image, image_path))

def _process_regression(path, image_list):
  """ Process images for regression. The label of each image is taken as the
  filename, without the extension.
  Args:
    path: The path to the images.
    image_list: The list containing tubles of the label, name, and path for all
                the images loaded so far. This will be added to. """
  for image in os.listdir(path):
    # Extract the label.
    label = os.path.splitext(image)[0]

    # Add the image.
    image_path = os.path.join(path, image)
    image_list.append((label, image, image_path))

def _build_dataset_and_cache(dataset_images, disk_cache, size, offset,
                             no_dataset=False, link_to=None):
  """ Builds a new dataset, and adds images to the disk cache.
  Args:
    dataset_images: The images to include in the dataset. These should be tuples
                    of the label, image name, and full image path.
    disk_cache: The DiskCache to add loaded images to.
    size: The x and y size of each image in the dataset.
    offset: The x and y offset to use when resizing each image.
    no_dataset:
      If true, it will ONLY build the cache. No dataset will be returned in this
      case.
    link_to: Path to a cache map to build a linked cache of, if specified.
  Returns:
    dataset: The dataset that was built from the data. """
  # Shuffle the images, and load them into the cache randomly. This is so that
  # contiguous loading from the cache actually works.
  print "Building image cache..."
  if not link_to:
    random.shuffle(dataset_images)
  else:
    # In this case, we want to sort from an existing cache.
    dataset_images = _sort_from_existing_cache(link_to, dataset_images)

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

  if not no_dataset:
    data = dataset.Dataset(dataset_entries, disk_cache, 0, (0, 0, 0))
    return data

def _sort_from_existing_cache(map_path, unsorted_images):
  """ Sorts a list of images into the same order as in an existing cache.
  Args:
    map_path: The path to the cache map that we are using as a template.
    unsorted_images: The raw images, of the form returned by
    collect_dataset_images. """
  print "Linking to existing cache '%s'." % (map_path)

  map_file = file(map_path, "rb")
  labels, offsets, free_start, free_end = pickle.load(map_file)
  map_file.close()

  # Get the first image.
  current_offset = 0 if free_start else free_end

  # Create a dictionary mapping image IDs to their position in the cache.
  image_positions = {}
  order_counter = 0
  while (len(image_positions) < len(offsets)):
    img_id = offsets[current_offset]

    # Add it the ordering dict.
    image_positions[img_id] = order_counter
    order_counter += 1

    # Go to the next one.
    label, name = utils.split_img_id(img_id)
    _, image_size = labels[label][name]
    current_offset += image_size

  # Now, re-order our images to match.
  ordered = [None] * len(image_positions)
  for image in unsorted_images:
    label, name, _ = image
    img_id = utils.make_img_id(label, name)

    # Find the correct position.
    position = image_positions[img_id]
    ordered[position] = image

  # Make sure everything matches.
  for i in range(0, len(ordered)):
    if ordered[i] == None:
      # Something didn't match up here.
      raise ValueError("No corresponding image at positon %d." % (i))

  return ordered

def collect_dataset_images(path, regression=False):
  """ Collects two lists of the images in the training and testing sets.
  Args:
    path: The location of the dataset.
    regression: Whether we are using regression. If so, we won't look for images
                broken up into discrete categories.
  Returns: A list of the images in the dataset. """
  print "Building dataset from images at %s..." % (path)

  if not os.path.exists(path):
    print "ERROR: I tried my best, but I couldn't find '%s'." % (path)
    print "I'm really sorry, old chap."
    sys.exit(1)

  dataset_images = []
  if not regression:
    for label in os.listdir(path):
      label_path = os.path.join(path, label)
      if not os.path.isdir(label_path):
        print "WARNING: Skipping extraneous item: '%s'" % (label_path)
        continue

      # Read the contents of the directory.
      _process_label(label, path, dataset_images)

  if regression:
    _process_regression(path, dataset_images)

  return dataset_images

def divide_sets(path, test_fraction=0.1, regression=False):
  """ Divides a set of images into a training set and testing set.
  Args:
    path: The path to the image directory. This should contain folders for
              each label.
    test_fraction: What fraction of the images should be used for a test set.
    regression: Whether we are doing a regression task.
  Returns: The list of training images, and the list of testing images.
  """
  # First, we'll load everything, and then split it later.
  all_images = collect_dataset_images(path, regression=regression)

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

def convert_dataset(location, size, output, seed, do_split=False,
                    offset=(0, 0), link_path=None, regression=False):
  """ Converts a dataset to a format that's usable by data_manager.
  Args:
    location: The location of the dataset to convert.
    size: The x and y sizes of each image in the dataset.
    output: The location to write output files to.
    seed: The seed to use for random number generation when splitting the
    batches.
    do_split: Whether to generate a new train/test split or use an existing one.
    offset: Optional offset to use for width and height when resizing the image.
    link_path: Optional existing cache_map file. If provided, it will only
               generate a cache in the same ordering as the one indicated.
    regression: Whether we're doing a regression task and the images are not in
                discrete categories.
  """
  random.seed(seed)

  # The conversion works by reading all the images and adding them to a
  # DiskCache.
  disk_cache = cache.DiskCache(output)

  if link_path:
    # Skip dataset building entirely and just build the cache.
    all_images = collect_dataset_images(location, regression=regression)
    _build_dataset_and_cache(all_images, disk_cache, size, offset,
                             no_dataset=True, link_to=link_path)

    return

  elif not do_split:
    # Individual categories are contained in the train and test directories.
    train_path = os.path.join(location, "train")
    test_path = os.path.join(location, "test")

    train_images = collect_dataset_images(train_path, regression=regression)
    test_images = collect_dataset_images(test_path, regression=regression)

  else:
    # We'll have to generate a train/test split.
    train_images, test_images = divide_sets(location, regression=regression)

  train_set = _build_dataset_and_cache(train_images, disk_cache, size, offset)
  test_set = _build_dataset_and_cache(test_images, disk_cache, size, offset)

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
  parser.add_argument("-r", "--random_seed", default=time.time(), type=int,
                      help="Specify random seed for batch shuffling.")
  parser.add_argument("-l", "--link", default=None,
                      help="Path to a cache_map file that we should link to.")
  parser.add_argument("-g", "--regression", action="store_true",
                      help="Whether to generate a regression dataset.")
  args = parser.parse_args()

  convert_dataset(args.dataset, (args.width, args.height), args.output,
                  args.random_seed, do_split=args.split_train_test,
                  offset=(args.offset_width, args.offset_height),
                  link_path=args.link, regression=args.regression)


if __name__ == "__main__":
  main()
