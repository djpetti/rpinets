""" A very simple FCFF NN intended to be used for comparing tensorflow to other
libraries. """


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
    """ Initializes tensors containing the weights for each layer.
    Args:
      layers: A list denoting the number of inputs of each layer.
      outputs: The number of outputs of the network. """
    self.__weights = []
    # This is in case we have a single hidden layer.
    fan_out = layers[0]
    for i in range(0, len(layers) - 1):
      fan_in = layers[i]
      fan_out = layers[i + 1]

      # Initialize weights randomly.
      weights = tf.Variable(tf.random_normal([fan_in, fan_out], stddev=0.35))
      self.__weights.append(weights)

    # Include outputs also.
    self.__weights.append(tf.Variable(tf.random_normal([fan_out, outputs])))

  def __add_layers(self, first_inputs):
    """ Adds as many hidden layers to our model as there are elements in
    __weights.
    Args:
      first_inputs: The tensor to use as inputs to the first hidden layer. """
    # Outputs from the previous layer that get used as inputs for the next
    # layer.
    next_inputs = first_inputs
    for weights in self.__weights:
      next_inputs = tf.nn.sigmoid(tf.matmul(next_inputs, weights))

    self._layer_stack = next_inputs

  def __build_model(self, layers, outputs):
    """ Actually constructs the graph for this model.
    Args:
      layers: A list denoting the number of inputs of each layer.
      outputs: The number of outputs of the network. """
    # Initialize all the weights first.
    self.__initialize_weights(layers, outputs)

    # Inputs and outputs.
    num_inputs = layers[0]
    self._inputs = tf.placeholder("float", [None, num_inputs])
    self._expected_outputs = tf.placeholder("float", [None, outputs])

    # Build actual layer model.
    self.__add_layers(self._inputs)

    # Cost function.
    cost = tf.reduce_mean( \
        tf.nn.softmax_cross_entropy_with_logits(self._layer_stack,
                                                self._expected_outputs))
    # SGD optimizer.
    self._optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    # Does an actual prediction.
    self._prediction_operation = tf.argmax(self._layer_stack, 1)

  def _extend_with_feedforward(self, inputs, layers, outputs):
    """ Meant to be used by subclasses as a simple way to extend the graph of a
    feedforward network. You pass in the inputs you want to use to create the
    feedforward network, and it constructs the network around that.
    Args:
      inputs: The inputs to use for the feedforward network.
      layers: A list denoting the number of inputs of each layer.
      outputs: The number of outputs of the network. """
    self.__initialize_weights(layers, outputs)
    self.__add_layers(inputs)

  def expected_outputs(self):
    """ Set the value of the outputs we expect for this cycle. This should be
    used to get one of the key values for the feed_dict argument of
    Session.run().
    Returns:
      The key value for feed_dict for the expected outputs. """
    return self._expected_outputs

  def inputs(self):
    """ Set the value of the outputs we expect for this cycle. This should be
    used to get the other key value for the feed_dict argument of Session.run().
    Returns:
      The key value for feed_dict for the network inputs. """
    return self._inputs

  def predict(self):
    """ Runs an actual prediction step for the network. It is intended that
    the result here get passed as the target of Session.run().
    Returns:
      The prediction operation. """
    return self._prediction_operation

  def train(self):
    """ Runs an SGD training step for the network. It is intended that the
    result here get passed as the target of Session.run().
    Returns:
      The training operation. """
    return self._optimizer
