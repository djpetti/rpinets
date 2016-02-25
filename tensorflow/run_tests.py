#!/usr/bin/python


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import numpy as np

from simple_lenet import LeNetClassifier


def run_mnist_test():
  """ Runs the a test that trains a CNN to classify MNIST digits. """
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  conv1 = LeNetClassifier.ConvLayer(kernel_width=3, kernel_height=3,
                                    feature_maps=1)
  conv2 = LeNetClassifier.ConvLayer(kernel_width=3, kernel_height=3,
                                    feature_maps=32)
  conv3 = LeNetClassifier.ConvLayer(kernel_width=3, kernel_height=3,
                                    feature_maps=64)
  network = LeNetClassifier((28, 28, 1), [conv1, conv2, conv3], [4 * 4 * 128, 625], 10)

  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)

  writer = tf.train.SummaryWriter("mnist_logs", sess.graph_def)

  for i in range(20000):
    batch = mnist.train.next_batch(128)
    if i%100 == 0:
      result = sess.run(network.predict(),
          feed_dict={network.inputs(): batch[0],
                    network.expected_outputs(): batch[1]})
      argmax = np.argmax(batch[1], axis=1)
      print("step %d, training accuracy %s"%(i, np.mean(argmax == result)))
    sess.run(network.train(), feed_dict={network.inputs(): batch[0],
                                        network.expected_outputs(): batch[1]})

def main():
  run_mnist_test()


if __name__ == "__main__":
  main()
