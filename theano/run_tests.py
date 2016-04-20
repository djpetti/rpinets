#!/usr/bin/python

import json
import os
import time

from alexnet import AlexNet
from simple_lenet import LeNetClassifier
import data_loader
import layers

from six.moves import cPickle as pickle


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

  conv1 = layers.ConvLayer(kernel_width=5, kernel_height=5, feature_maps=1)
  conv2 = layers.ConvLayer(kernel_width=3, kernel_height=3, feature_maps=32)
  pool = layers.PoolLayer()
  inner_product1 = layers.InnerProductLayer(size=5 * 5 * 128)
  inner_product2 = layers.InnerProductLayer(size=625)
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

    cost = network.train(train_batch_index)[0]
    if iterations % 100 == 0:
      print "Training cost: %f" % (cost)

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
  load_batches = 1

  # Learning rate hyperparameters.
  learning_rate = 0.001
  decay_steps = 10000
  decay_rate = 1
  momentum = 0.9
  weight_decay = 0.0005

  rho = 0.9
  epsilon = 1e-6

  # Where we save the network.
  save_file = "alexnet.pkl"
  synsets_save_file = "synsets.pkl"

  data = data_loader.Ilsvrc12(batch_size, load_batches, use_4d=True)
  if os.path.exists(synsets_save_file):
    data.load(synsets_save_file)
  train = data.get_train_set()
  test = data.get_test_set()
  _, cpu_labels = data.get_non_shared_test_set()

  if os.path.exists(save_file):
    # Load from the file.
    print "Theano: Loading network from file..."
    network = AlexNet.load(save_file, train, test, batch_size,
                           learning_rate=learning_rate)

  else:
    # Build new network.
    conv1 = layers.ConvLayer(kernel_width=11, kernel_height=11, stride_width=4,
                             stride_height=4, feature_maps=3,
                             border_mode="half")
    conv2 = layers.ConvLayer(kernel_width=5, kernel_height=5, feature_maps=96,
                             border_mode="half", start_bias=1)
    conv3 = layers.ConvLayer(kernel_width=3, kernel_height=3, feature_maps=256,
                             border_mode="half")
    conv4 = layers.ConvLayer(kernel_width=3, kernel_height=3, feature_maps=384,
                             border_mode="half", start_bias=1)
    conv5 = layers.ConvLayer(kernel_width=3, kernel_height=3, feature_maps=384,
                             border_mode="half", start_bias=1)
    pool = layers.PoolLayer(kernel_width=3, kernel_height=3, stride_width=2,
                            stride_height=2)
    flatten = layers.InnerProductLayer(size=6 * 6 * 256, dropout=True,
                                       start_bias=1)
    inner_product1 = layers.InnerProductLayer(size=4096, dropout=True,
                                              start_bias=1)
    inner_product2 = layers.InnerProductLayer(size=4096)
    norm = layers.NormalizationLayer(depth_radius=5, alpha=1e-05 ,beta=0.75,
                                     bias=1.0)
    network = AlexNet((224, 224, 3), [conv1, pool, norm, conv2, pool,
                                      norm, conv3, conv4, conv5, pool,
                                      flatten, inner_product1, inner_product2],
                      1000, train, test, batch_size)

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

  while iterations < 800000:
    if iterations % 50 == 0:
      # FIXME (danielp): Another hack for dealing with VRAM storage. We test in
      # five parts, with each set of batches loaded individually.
      for _ in range(0, 4):
        if not test:
          # Test data is not loaded.
          test = data.get_test_set()
          _, cpu_labels = data.get_non_shared_test_set()
        print cpu_labels
        network.test_part(test_batch_index, cpu_labels)
        test = None
      test = data.get_test_set()
      _, cpu_labels = data.get_non_shared_test_set()
      top_one, top_five = network.test(test_batch_index, cpu_labels)
      test = None
      print "Theano: step %d, testing top 1: %f, testing top 5: %f" % \
            (iterations, top_one, top_five)

      test_batch_index += 1

    cost = network.train(train_batch_index)[0]
    print "Training cost: %f" % (cost)

    if iterations % 50 == 0:
      print "Saving network..."
      network.save(save_file)
      # Save synset data as well.
      data.save(synsets_save_file)

    iterations += 1
    train_batch_index += 1

    # Swap in new data if we need to.
    if (train_batch_index + 1) * batch_size > data.get_train_set_size():
      train_batch_index = 0
      train = data.get_train_set()
    # Swap in new data if we need to.
    if (test_batch_index + 1) * batch_size > data.get_test_set_size():
      test_batch_index = 0

  elapsed = time.time() - start_time
  speed = iterations / elapsed
  print("Theano: Ran %d training iterations. (%f iter/s)" % \
      (iterations, speed))
  print("Theano: Imagenet test completed in %f seconds." % (elapsed))
  return (elapsed, speed)

def main():
  elapsed, speed = run_imagenet_test()
  results = {"mnist": {"elapsed": elapsed, "speed": speed}}
  print "results=%s" % (json.dumps(results))


if __name__ == "__main__":
  main()
