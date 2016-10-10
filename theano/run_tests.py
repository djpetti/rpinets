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

import json
import os
import time

from alexnet import AlexNet
from simple_lenet import LeNetClassifier
import layers

from six.moves import cPickle as pickle


logger = logging.getLogger(__name__)


def run_mnist_test():
  """ Runs the a test that trains a CNN to classify MNIST digits.
  Returns:
    A tuple containing the total elapsed time, and the average number of
    training iterations per second. """
  data = data_loader.Mnist(use_4d=True)
  train = data.get_train_set()
  test = data.get_test_set()

  batch_size = 128

  learning_rate = 0.01
  rho = 0.9
  epsilon = 1e-6

  conv1 = layers.ConvLayer(kernel_width=5, kernel_height=5, feature_maps=32)
  conv2 = layers.ConvLayer(kernel_width=3, kernel_height=3, feature_maps=128)
  pool = layers.PoolLayer()
  inner_product1 = layers.InnerProductLayer(size=5 * 5 * 128,
                                            weight_init="gaussian",
                                            weight_stddev=0.005)
  inner_product2 = layers.InnerProductLayer(size=625,
                                            weight_init="gaussian",
                                            weight_stddev=0.005)
  network = LeNetClassifier((28, 28, 1), [conv1, pool, conv2, pool,
                                          inner_product1, inner_product2],
                            10, train, test, batch_size)

  network.use_rmsprop_trainer(learning_rate, rho, epsilon)

  print("Theano: Starting MNIST test...")

  accuracy = 0
  start_time = time.time()
  iterations = 0

  train_batch_index = 0
  test_batch_index = 0

  while iterations < 2000:
    if iterations % 500 == 0:
      accuracy = network.test(test_batch_index)
      print("Tensorflow: step %d, testing accuracy %s" % \
            (iterations, accuracy))

      test_batch_index += 1

    cost, rate, step = network.train(train_batch_index)
    if iterations % 100 == 0:
      print "Training cost: %f, learning rate: %f, step: %d" % \
            (cost, rate, step)

    iterations += 1
    train_batch_index += 1

    # Wrap indices.
    if (train_batch_index + 1) * batch_size >= data.get_train_set_size():
      train_batch_index = 0
    if (test_batch_index + 1) * batch_size >= data.get_test_set_size():
      test_batch_index = 0

  elapsed = time.time() - start_time
  speed = iterations / elapsed
  print("Theano: Ran %d training iterations. (%f iter/s)" % \
      (iterations, speed))
  print("Theano: MNIST test completed in %f seconds." % (elapsed))
  return (elapsed, speed)

def run_imagenet_test():
  """ Runs the a test that trains a CNN to classify ImageNet data.
  Returns:
    A tuple containing the total elapsed time, and the average number of
    training iterations per second. """
  batch_size = 128
  # How many batches to have loaded into VRAM at once.
  load_batches = 5

  # Learning rate hyperparameters.
  learning_rate = 0.01
  decay_steps = 10000
  decay_rate = 1
  momentum = 0.9
  weight_decay = 0.0005

  rho = 0.9
  epsilon = 1e-6

  # Where we save the network.
  save_file = "/home/theano/training_data/alexnet.pkl"
  synsets_save_file = "/home/theano/training_data/synsets.pkl"

  data = data_loader.Ilsvrc12(batch_size, load_batches)
  if os.path.exists(synsets_save_file):
    data.load(synsets_save_file)
  train = data.get_train_set()
  test = data.get_test_set()
  cpu_labels = data.get_non_shared_test_set()

  if os.path.exists(save_file):
    # Load from the file.
    print "Theano: Loading network from file..."
    network = AlexNet.load(save_file, train, test, batch_size,
                           learning_rate=learning_rate)

  else:
    # Build new network.
    network = AlexNet(train, test, batch_size,
                      patch_separation=batch_size * load_batches)

    network.use_sgd_trainer(learning_rate, momentum=momentum,
                            weight_decay=weight_decay,
                            decay_rate=decay_rate,
                            decay_steps=decay_steps)
    #network.use_rmsprop_trainer(learning_rate, rho, epsilon,
    #                            decay_rate=decay_rate,
    #                            decay_steps=decay_steps)

  print "Theano: Starting ImageNet test..."

  accuracy = 0
  start_time = time.time()
  iterations = 0

  train_batch_index = 0
  test_batch_index = 0

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
    test_set_one_patch = data.get_test_set_size() / 10
    if (test_batch_index + 1) * batch_size > test_set_one_patch:
      test_batch_index = 0
      logging.info("Getting test set.")
      test = data.get_test_set()
      cpu_labels = data.get_non_shared_test_set()[:]
      logger.info("Got test set.")

    if iterations % 100 == 0:
      # cpu_labels contains labels for every batch currently loaded in VRAM,
      # without duplicates for additional patches.
      label_index = test_batch_index * batch_size
      top_one, top_five = network.test(test_batch_index,
                                       cpu_labels[label_index:label_index + \
                                                              batch_size])
      print "Theano: step %d, testing top 1: %f, testing top 5: %f" % \
            (iterations, top_one, top_five)

      test_batch_index += 1

    cost, rate, step = network.train(train_batch_index)
    print "Training cost: %f, learning rate: %f, step: %d" % \
            (cost, rate, step)

    if iterations % 50 == 0:
      print "Saving network..."
      network.save(save_file)
      # Save synset data as well.
      data.save(synsets_save_file)

    iterations += 1
    train_batch_index += 1

  elapsed = time.time() - start_time
  speed = iterations / elapsed
  print("Theano: Ran %d training iterations. (%f iter/s)" % \
      (iterations, speed))
  print("Theano: Imagenet test completed in %f seconds." % (elapsed))

  data.exit_gracefully()

  return (elapsed, speed)

def evaluate_final_alexnet():
  """ Quick way to evaluate AlexNet performance once it's done training. """
  # Where we save the network.
  save_file = "alexnet.pkl"
  synsets_save_file = "synsets.pkl"

  batch_size = 128
  load_batches = 1

  data = data_loader.Ilsvrc12(batch_size, load_batches, use_4d=True)
  data.load(synsets_save_file)

  test = data.get_test_set()
  train = data.get_train_set()
  _, cpu_labels = data.get_non_shared_test_set()

  # Load from the file.
  print "Theano: Loading network from file..."
  network = AlexNet.load(save_file, train, test, batch_size)
  print "Done."

  # Test on a fairly large sample.
  total_one = 0
  total_five = 0
  test_batch_index = 0
  for _ in range(0, 10):
    # FIXME (danielp): Another hack for dealing with VRAM storage. We test in
    # five parts, with each set of batches loaded individually.
    for _ in range(0, 4):
      if not test:
        # Test data is not loaded.
        test = data.get_test_set()
        _, cpu_labels = data.get_non_shared_test_set()
      network.test_part(test_batch_index, cpu_labels)
      test = None
    test = data.get_test_set()
    _, cpu_labels = data.get_non_shared_test_set()
    top_one, top_five = network.test(test_batch_index, cpu_labels)
    test = None
    print "Theano: testing top 1: %f, testing top 5: %f" % \
          (top_one, top_five)

    total_one += top_one
    total_five += top_five

  average_one = total_one / 10.0
  average_five = total_five / 10.0
  print "Theano: Mean top 1: %f, mean top 5: %f" % \
        (average_one, average_five)

def main():
  elapsed, speed = run_imagenet_test()
  results = {"imagenet": {"elapsed": elapsed, "speed": speed}}
  print "results=%s" % (json.dumps(results))

if __name__ == "__main__":
  main()
