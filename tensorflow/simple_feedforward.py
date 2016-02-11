#!/usr/bin/python

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import numpy as np


class FeedforwardNetwork(object):
  """ A simple, fully-connected feedforward neural network. """

  def __init__(self, layers, outputs):
    """
    Args:
      layers: A list of ints denoting the number of inputs for each layer. It
      is assumed that the outputs of one layer will be the same size as the
      inputs of the next one.
      outputs: The number of outputs of the network. """
    self.__build_model(layers, outputs)

  def __initialize_weights(self, layers, outputs):
    """ Initializes tensors containing the wieghts for each layer.
    Args:
      layers: A list denoting the number of inputs of each layer.
      outputs: The number of outputs of the network. """
    self.__weights = []
    for i in range(0, len(layers) - 1):
      fan_in = layers[i]
      fan_out = layers[i + 1]

      # For good performance, the standard is to initialize weights randomly,
      # but we initialize them to zero since we're seeking better repeatability.
      weights = tf.Variable(tf.random_normal([fan_in, fan_out], stddev=0.35))
      self.__weights.append(weights)

    # Include outputs also.
    self.__weights.append(tf.Variable(tf.zeros([fan_out, outputs])))

  def __add_layers(self):
    """ Adds as many hidden layers to our model as there are elements in
    __weights. """
    # Outputs from the previous layer that get used as inputs for the next
    # layer.
    next_inputs = self.__inputs
    for weights in self.__weights:
      next_inputs = tf.nn.sigmoid(tf.matmul(next_inputs, weights))

    self.__layer_stack = next_inputs

  def __build_model(self, layers, outputs):
    """ Actually constructs the graph for this model.
    Args:
      layers: A list denoting the number of inputs of each layer.
      outputs: The number of outputs of the network. """
    # Initialize all the weights first.
    self.__initialize_weights(layers, outputs)

    # Inputs and outputs.
    num_inputs = layers[0]
    self.__inputs = tf.placeholder("float", [None, num_inputs])
    self.__expected_outputs = tf.placeholder("float", [None, outputs])

    # Build actual layer model.
    self.__add_layers()

    # Cost function.
    cost = tf.reduce_mean( \
        tf.nn.softmax_cross_entropy_with_logits(self.__layer_stack,
                                                self.__expected_outputs))
    # SGD optimizer.
    self.__optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    # Does an actual prediction.
    self.__prediction_operation = tf.argmax(self.__layer_stack, 1)

  def expected_outputs(self):
    """ Set the value of the outputs we expect for this cycle. This should be
    used to get one of the key values for the feed_dict argument of
    Session.run().
    Returns:
      The key value for feed_dict for the expected outputs. """
    return self.__expected_outputs

  def inputs(self):
    """ Set the value of the outputs we expect for this cycle. This should be
    used to get the other key value for the feed_dict argument of Session.run().
    Returns:
      The key value for feed_dict for the network inputs. """
    return self.__inputs

  def predict(self):
    """ Runs an actual prediction step for the network. It is intended that
    the result here get passed as the target of Session.run().
    Returns:
      The prediction operation. """
    return self.__prediction_operation

  def train(self):
    """ Runs an SGD training step for the network. It is intended that the
    result here get passed as the target of Session.run().
    Returns:
      The training operation. """
    return self.__optimizer
