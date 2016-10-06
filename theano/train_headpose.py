#!/usr/bin/python

import logging

def _configure_logging():
  """ Configure logging handlers. """
  # Configure root logger.
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  file_handler = logging.FileHandler("/job_files/run_tests.log")
  file_handler.setLevel(logging.DEBUG)
  stream_handler = logging.StreamHandler()
  stream_handler.setLevel(logging.WARNING)
  formatter = logging.Formatter("%(name)s@%(asctime)s: " +
      "[%(levelname)s] %(message)s")
  file_handler.setFormatter(formatter)
  stream_handler.setFormatter(formatter)
  root.addHandler(file_handler)
  root.addHandler(stream_handler)

# Some modules need a logger to be configured immediately.
_configure_logging()

# This forks a lot of processes, so we want to import it as soon as possible,
# when there is as little memory as possible in use.
import data_loader

from simple_lenet import LeNetClassifier

from six.moves import cPickle as pickle
import os

import layers


logger = logging.getLogger(__name__)


# Batsh size
batch_size = 10
# How many batches to have loaded in VRAM at once.
load_batches = 50

# Learning rate specification.
learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0005

# Where we save the network.
save_file = "/job_files/headpose.pkl"
# Where to save label mapping information.
label_save_file = "/job_files/headpose_labels.pkl"
# Where are datasets are stored.
dataset_location = "/training_data/rpinets/headpose/dataset"


def main():
  # Data loading.
  data = data_loader.DataManagerLoader(batch_size, load_batches, (255, 255, 1),
                                       dataset_location=dataset_location)
  if os.path.exits(label_save_file):
    logger.info("Loading existing label mapping from %s..." % (label_save_file))
    data.load(label_save_file)
  train = data.get_train_set()
  test = data.get_test_set()

  # Initialize the network.
  if os.path.exists(save_file):
    # Load from the file.
    logger.info("Loading network from file: %s..." % (save_file))
    network = LeNetClassifier.load(save_file, train, test, batch_size,
                                   learning_rate=learning_rate)

  else:
    # Build a new network.
    logger.info("Building new network...")

    conv1 = layers.ConvLayer(kernel_width=5, kernel_height=3, feature_maps=3)
    conv2 = layers.ConvLayer(kernel_width=2, kernel_height=2, feature_maps=19)
    pool = layers.PoolLayer(kernel_width=2, kernel_height=2)
    inner_product1 = layers.InnerProductLayer(size=13, dropout=True,
                                              start_bias=1,
                                              weight_init="gaussian",
                                              weight_stddev=0.005)
    inner_product2 = layers.InnerProductLayer(size=10, dropout=True,
                                              start_bias=1,
                                              weight_init="gaussian",
                                              weight_stddev=0.005)
    use_layers = [conv1, conv2, pool, inner_product1, inner_product2]

    network = LeNetClassifier((225, 255, 1), use_layers, 2, train, test,
                              batch_size)

    network.use_sgd_trainer(learning_rate, momentum=momentum,
                            weight_decay=weight_decay)

  logger.info("Starting headpose training...")

  iterations = 0
  while iterations < 50000:
    logger.debug("Train index, size: %d, %d" % (train_batch_index,
                                                data.get_train_set_size()))
    logger.debug("Test index, size: %d, %d" % (test_batch_index,
                                               data.get_test_set_size()))

    # Swap in new data if we need to.
    if (train_batch_index + 1) * batch_size > data.get_train_set_size():
      train_batch_index = 0
      logging.info("Getting train set.")
      train = data.get_train_set()
      logging.info("Got train set.")
    # Swap in new data if we need to.
    if (test_batch_index + 1) * batch_size > data.get_test_set_size():
      test_batch_index = 0
      logging.info("Getting test set.")
      test = data.get_test_set()
      logging.info("Got test set.")

    if iterations % 50 == 0:
      cpu_labels = data.get_non_shared_test_set()
      logger.info("Finished loading test data.")

      label_index = test_batch_index
      top_one, top_five = network.test(test_batch_index,
                                       cpu_labels[label_index:label_index + \
                                                              batch_size])
      print "Theano: step %d, testing top 1: %f, testing top 5: %f" % \
            (iterations, top_one, top_five)

      # We have 10 translations, so each batch actually encompasses 10 times as
      # many images.
      test_batch_index += 1

    cost, rate, step = network.train(train_batch_index)
    print "Training cost: %f, learning rate: %f, step: %d" % \
            (cost, rate, step)

    if iterations % 50 == 0:
      print "Saving network..."
      network.save(save_file)
      # Save label data as well.
      data.save(label_save_file)

    iterations += 1
    train_batch_index += 1

