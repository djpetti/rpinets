#!/usr/bin/python

import json
import time

from simple_lenet import LeNetClassifier
from simple_feedforward import FeedforwardNetwork
import mnist


def run_mnist_test():
  """ Runs the a test that trains a CNN to classify MNIST digits.
  Returns:
    A tuple containing the total elapsed time, and the average number of
    training iterations per second. """
  data = mnist.Mnist(use_4d=False)
  train = data.get_train_set()
  test = data.get_test_set()

  batch_size = 128

  conv1 = LeNetClassifier.ConvLayer(kernel_width=5, kernel_height=5,
                                    feature_maps=1)
  conv2 = LeNetClassifier.ConvLayer(kernel_width=3, kernel_height=3,
                                    feature_maps=32)
  network = LeNetClassifier((28, 28, 1), [conv1, conv2],
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
  print("Tensorflow: Ran %d training iterations. (%f iter/s)" % \
      (iterations, speed))
  print("Tensorflow: MNIST test completed in %f seconds." % (elapsed))
  return (elapsed, speed)

def main():
  elapsed, speed = run_mnist_test()
  results = {"mnist": {"elapsed": elapsed, "speed": speed}}
  print "results=%s" % (json.dumps(results))


if __name__ == "__main__":
  main()
