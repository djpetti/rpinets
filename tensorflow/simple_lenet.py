#!/usr/bin/python

""" A very simple LeNet implementation intended to be used for comparing
tensorflow to other libraries. """

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import numpy as np

from simple_feedforward import FeedforwardNetwork


class LeNetClassifier(FeedforwardNetwork):
  """ A classifier built upon the Convolutional Neural Network as described by
  Yan LeCun. """

  class ConvLayer(object):
    """ A simple class to handle the specification of convolutional layers. """

    def __init__(self, *args, **kwargs):
      # Convolutional kernel width.
      self.kernel_width = kwargs.get("kernel_width")
      # Convolutional kernel height.
      self.kernel_height = kwargs.get("kernel_height")
      # Number of input feature maps.
      self.feature_maps = kwargs.get("feature_maps")

  def __init__(self, image_size, conv_layers, feedforward_layers, outputs):
    """
    Args:
      image_size: Size of the image. (width, height, channels)
      conv_layers: A list of convolutional layers, composed of ConvLayer
      instances.
      feedforward_layers: A list of ints denoting the number of inputs for each
      fully-connected layer.
      outputs: The number of outputs of the network. """
    # We don't call the base class constructor here, because we want it to build
    # its network on top of our convolutional part.
    self.__build_model(image_size, conv_layers, feedforward_layers, outputs)

  def __initialize_weights(self, image_size, conv_layers, feedforward_inputs):
    """ Initializes tensors containing the weights for each convolutional layer.
    Args:
      image_size: The size of the input image.
      conv_layers: A list of ConvLayer instances describing all the
      convolutional layers.
      feedforward_inputs: The number of inputs in the first feedforward layer. """
    self.__weights = []
    for i in range(0, len(conv_layers) - 1):
      first_layer = conv_layers[i]
      next_layer = conv_layers[i + 1]

      # Initialize weights randomly.
      shape = [first_layer.kernel_width, first_layer.kernel_height,
               first_layer.feature_maps, next_layer.feature_maps]
      weights = tf.Variable(tf.random_normal(shape, stddev=0.35))
      self.__weights.append(weights)

    # Calculate our final input size based on the number of maxpoolings we do.
    image_x, image_y, channels = image_size
    final_x = image_x / (2 * len(conv_layers))
    final_y = image_y / (2 * len(conv_layers))

    # Add last convolutional layer weights.
    shape = [next_layer.kernel_width, next_layer.kernel_height,
             next_layer.feature_maps, feedforward_inputs / \
                final_x / final_y / channels]
    weights = tf.Variable(tf.random_normal(shape, stddev=0.35))
    self.__weights.append(weights)

  def __add_layers(self, feedforward_layers, outputs):
    """ Adds as many convolutional layers to our model as there are elements in
    __weights.
    Args:
      feedforward_layers: A list denoting the number of inputs for each
      feedforward layer.
      outputs: The number of outputs of the network. """
    # Outputs from the previous layer that get used as inputs for the next
    # layer.
    next_inputs = self.__reshaped_inputs
    for weights in self.__weights:
      # Convolution.
      conv = tf.nn.conv2d(next_inputs, weights, strides=[1, 1, 1, 1],
                          padding="SAME")
      # Activation.
      conv = tf.nn.relu(conv)
      # Max pooling.
      next_inputs = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding="SAME")

    # Reshape convolution outputs so they can be used as inputs to the
    # feedforward network.
    num_inputs = feedforward_layers[0]
    flattened_inputs = tf.reshape(next_inputs, [-1, num_inputs])
    # Build the fully-connected part of the network.
    self._extend_with_feedforward(flattened_inputs, feedforward_layers, outputs)

  def __build_model(self, image_size, conv_layers, feedforward_layers, outputs):
    """ Constructs the graph for this model.
    Args:
      image_size: Size of the image.
      conv_layers: A list of ConvLayer instances describing all the
      convolutional layers.
      feedforward_layers: A list denoting the number of inputs for each
      feedforward layer.
      ouputs: The number of outputs of the network. """
    # Initialize all the weights first.
    num_inputs = feedforward_layers[0]
    self.__initialize_weights(image_size, conv_layers, num_inputs)

    # Inputs and outputs.
    self._inputs = tf.placeholder("float", [None, None])
    self._expected_outputs = tf.placeholder("float", [None, outputs])
    # Reshape inputs to a 4D tensor.
    new_shape = [-1] + list(image_size)
    self.__reshaped_inputs = tf.reshape(self._inputs, new_shape)

    # Build actual layer model.
    self.__add_layers(feedforward_layers, outputs)

    # Now _layer_stack should contain the entire network.
    # Build cost function.
    cost = tf.reduce_mean( \
        tf.nn.softmax_cross_entropy_with_logits(self._layer_stack,
                                                self._expected_outputs))
    # SGD optimizer.
    self._optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    # Does an actual prediction.
    self._prediction_operation = tf.argmax(self._layer_stack, 1)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

conv1 = LeNetClassifier.ConvLayer(kernel_width=3, kernel_height=3,
                                  feature_maps=1)
conv2 = LeNetClassifier.ConvLayer(kernel_width=3, kernel_height=3,
                                  feature_maps=32)
network = LeNetClassifier((28, 28, 1), [conv1, conv2], [7 * 7 * 512], 10)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    result = sess.run(network.predict(),
        feed_dict={network.inputs(): batch[0],
                   network.expected_outputs(): batch[1]})
    argmax = np.argmax(batch[1], axis=1)
    print("step %d, training accuracy %s"%(i, np.mean(argmax == result)))
  sess.run(network.train(), feed_dict={network.inputs(): batch[0],
                                       network.expected_outputs(): batch[1]})
