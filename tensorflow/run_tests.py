#!/usr/bin/python

import json
import time

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import numpy as np

from simple_lenet import LeNetClassifier


def run_mnist_test():
  """ Runs the a test that trains a CNN to classify MNIST digits.
  Returns:
    A tuple containing the total elapsed time, and the average number of
    training iterations per second. """
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  train_x, train_y = mnist.train.images, mnist.train.labels,
  test_x, test_y = mnist.test.images, mnist.test.labels
  # Reshape right off the bat to save some time.
  train_x = train_x.reshape(-1, 28, 28, 1)
  test_x = test_x.reshape(-1, 28, 28, 1)

  conv1 = LeNetClassifier.ConvLayer(kernel_width=5, kernel_height=5,
                                    feature_maps=1)
  conv2 = LeNetClassifier.ConvLayer(kernel_width=3, kernel_height=3,
                                    feature_maps=32)
  conv3 = LeNetClassifier.ConvLayer(kernel_width=3, kernel_height=3,
                                    feature_maps=64)
  network = LeNetClassifier((28, 28, 1), [conv1, conv2, conv3],
                            [4 * 4 * 128, 625], 10, batch_size=128)

  saver = tf.train.Saver()

  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)

  writer = tf.train.SummaryWriter("mnist_logs", sess.graph_def)

  print("Tensorflow: Starting MNIST test...")

  accuracy = 0
  start_time = time.time()
  iterations = 0
  while iterations < 2000:
    if iterations % 500 == 0:
      test_batch = mnist.test.next_batch(128)
      result = sess.run(network.predict(),
          feed_dict={network.inputs(): test_batch[0],
                    network.expected_outputs(): test_batch[1]})
      argmax = np.argmax(test_batch[1], axis=1)
      accuracy = np.mean(argmax == result)
      print("Tensorflow: step %d, testing accuracy %s" % \
            (iterations, accuracy))

    batch = mnist.train.next_batch(128)
    sess.run(network.train(), feed_dict={network.inputs(): batch[0],
                                        network.expected_outputs(): batch[1]})
    iterations += 1

  # Save the network at the end.
  #saver.save(sess, "Variables/test.ckpt")

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
