#!/usr/bin/python

import json
import time

import data_loader
from simple_lenet import LeNetClassifier
from simple_feedforward import FeedforwardNetwork


def run_mnist_test():
  """ Runs the a test that trains a CNN to classify MNIST digits.
  Returns:
    A tuple containing the total elapsed time, and the average number of
    training iterations per second. """
  data = data_loader.Mnist(use_4d=True)
  train = data.get_train_set()
  test = data.get_test_set()

  batch_size = 128

  conv1 = LeNetClassifier.ConvLayer(kernel_width=5, kernel_height=5,
                                    feature_maps=1)
  conv2 = LeNetClassifier.ConvLayer(kernel_width=3, kernel_height=3,
                                    feature_maps=32)
  pool = LeNetClassifier.PoolLayer()
  network = LeNetClassifier((28, 28, 1), [conv1, pool, conv2, pool],
                            [5 * 5 * 128, 625], 10, train, test, batch_size)
  #network = FeedforwardNetwork([784, 625], 10, train, test, batch_size)

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

  data = data_loader.Ilsvrc12(batch_size, load_batches, use_4d=True)
  train = data.get_train_set()
  test = data.get_test_set()

  conv1 = LeNetClassifier.ConvLayer(kernel_width=11, kernel_height=11,
                                    stride_width=4, stride_height=4,
                                    feature_maps=3)
  pool1 = LeNetClassifier.PoolLayer(kernel_width=3, kernel_height=3,
                                    stride_width=2, stride_height=2)
  conv2 = LeNetClassifier.ConvLayer(kernel_width=5, kernel_height=5,
                                    feature_maps=96)
  pool2 = LeNetClassifier.PoolLayer(kernel_width=3, kernel_height=3,
                                    stride_width=2, stride_height=2)
  conv3 = LeNetClassifier.ConvLayer(kernel_width=3, kernel_height=3,
                                    feature_maps=256)
  conv4 = LeNetClassifier.ConvLayer(kernel_width=3, kernel_height=3,
                                    feature_maps=384)
  conv5 = LeNetClassifier.ConvLayer(kernel_width=3, kernel_height=3,
                                    feature_maps=384)
  pool5 = LeNetClassifier.PoolLayer(kernel_width=3, kernel_height=3,
                                    stride_width=2, stride_height=2)
  network = LeNetClassifier((28, 28, 1), [conv1, pool1, conv2, pool2, conv3,
                                          conv4, conv5, pool5],
                            [4096, 4096], 1000, train, test, batch_size)

  print("Theano: Starting ImageNet test...")

  accuracy = 0
  start_time = time.time()
  iterations = 0

  train_batch_index = 0
  test_batch_index = 0

  while iterations < 2000:
    if iterations % 50 == 0:
      accuracy = network.test(test_batch_index)
      print("Tensorflow: step %d, testing accuracy %s" % \
            (iterations, accuracy))

      test_batch_index += 1

    cost = network.train(train_batch_index)[0]
    if iterations % 10 == 0:
      print "Training cost: %f" % (cost)

    iterations += 1
    train_batch_index += 1

    # Swap in new data if we need to.
    if (train_batch_index + 1) * batch_size >= data.get_train_set_size():
      train_batch_index = 0
      train = data.get_train_set()
    if (test_batch_index + 1) * batch_size >= data.get_test_set_size():
      test_batch_index = 0
      test = data.get_test_set()

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
