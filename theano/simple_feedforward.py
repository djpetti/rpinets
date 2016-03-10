""" A very simple FCFF NN intended to be used for comparing theano to other
libraries. """


import theano
import theano.tensor as TT

import numpy as np


class FeedforwardNetwork(object):
  """ A simple, fully-connected feedforward neural network. """

  def __init__(self, layers, outputs, train, test, batch_size):
    """
    Args:
      layers: A list of ints denoting the number of inputs for each layer. It
      is assumed that the outputs of one layer will be the same size as the
      inputs of the next one.
      outputs: The number of outputs of the network.
      train: Training dataset, should be pair of inputs and expected outputs.
      test: Testing dataset, should be pair of inputs and expected outputs.
      batch_size: The size of each minibatch. """
    self._train_x, self._train_y = train
    self._test_x, self._test_y = test
    self._batch_size = batch_size

    self.__print_op = theano.printing.Print("Debug print: ")

    self.__build_model(layers, outputs)

  def __initialize_weights(self, layers, outputs):
    """ Initializes tensors containing the weights for each layer.
    Args:
      layers: A list denoting the number of inputs of each layer.
      outputs: The number of outputs of the network. """
    self.__weights = []
    # Keeps track of weight shapes because Theano is annoying about that.
    self.__weight_shapes = []
    # This is in case we have a single hidden layer.
    fan_out = layers[0]
    for i in range(0, len(layers) - 1):
      fan_in = layers[i]
      fan_out = layers[i + 1]

      # Initialize weights randomly.
      weights_values = np.random.normal(size=(fan_in, fan_out))
      weights = theano.shared(weights_values)
      self.__weights.append(weights)
      self.__weight_shapes.append((fan_in, fan_out))

    # Include outputs also.
    weights_values = np.random.normal(size=(fan_out, outputs))
    self.__weights.append(theano.shared(weights_values))
    self.__weight_shapes.append((fan_out, outputs))

  def __add_layers(self, first_inputs):
    """ Adds as many hidden layers to our model as there are elements in
    __weights.
    Args:
      first_inputs: The tensor to use as inputs to the first hidden layer. """
    # Outputs from the previous layer that get used as inputs for the next
    # layer.
    self.__biases = []
    next_inputs = first_inputs
    for i in range(0, len(self.__weights)):
      weights = self.__weights[i]
      _, fan_out = self.__weight_shapes[i]

      bias_values = np.zeros((fan_out,), dtype=theano.config.floatX)
      bias = theano.shared(bias_values)
      self.__biases.append(bias)
      sums = TT.dot(next_inputs, weights) + bias
      if i < len(self.__weights) - 1:
        next_inputs = TT.nnet.relu(sums)
      else:
        # For the last layer, we don't use an activation function.
        next_inputs = sums

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
    self._inputs = TT.matrix("inputs")
    self._expected_outputs = TT.ivector("expected_outputs")

    # Build actual layer model.
    self.__add_layers(self._inputs)

    # Cost function.
    cost = TT.mean( \
        self._softmax_cross_entropy_with_logits(self._layer_stack,
                                                self._expected_outputs))
    # SGD optimizer.
    self._optimizer = self._build_sgd_trainer(cost, 0.05, self._train_x,
                                              self._train_y, self._batch_size)

    # Does an actual prediction.
    self._prediction_operation = self._build_predictor(self._test_x,
                                                       self._batch_size)
    # Evaluates the network's accuracy on the testing data.
    self._tester = self._build_tester(self._test_x, self._test_y,
                                      self._batch_size)

  def _build_sgd_trainer(self, cost, learning_rate, train_x, train_y,
                         batch_size):
    """ Builds a new SGD trainer for the network.
    Args:
      cost: The cost function we are using.
      learning_rate: The learning rate to use for training.
      train_x: Training set inputs.
      train_y: Training set expected outputs.
      batch_size: How big our batches are.
    Returns:
      Theano function for training the network. """
    # Compute gradients for all parameters.
    params = self.__weights + self.__biases
    gradients = [TT.grad(cost, param) for param in params]

    # Tell it how to update the parameters.
    updates = []
    for param, gradient in zip(params, gradients):
      updates.append((param, param - learning_rate * gradient))

    # Index to a minibatch.
    index = TT.lscalar()
    # Create the actual function.
    batch_start = index * batch_size
    batch_end = (index + 1) * batch_size
    trainer = theano.function(inputs=[index], outputs=cost, updates=updates,
                              givens={self._inputs: \
                                      train_x[batch_start:batch_end],
                                      self._expected_outputs: \
                                      train_y[batch_start:batch_end]})
    return trainer

  def _build_predictor(self, test_x, batch_size):
    """ Builds a prediction function that computes a forward pass.
    Args:
      test_x: Testing set inputs.
      batch_size: How big our batches are.
    Returns:
      Theano function for testing the network. """
    index = TT.lscalar()
    outputs = TT.argmax(self._layer_stack, axis=1)

    batch_start = index * batch_size
    batch_end = (index + 1) * batch_size
    predictor = theano.function(inputs=[index], outputs=outputs,
                                givens={self._inputs: \
                                        test_x[batch_start:batch_end]})
    return predictor

  def _build_tester(self, test_x, test_y, batch_size):
    """ Builds a function that can be used to evaluate the accuracy of the
    network on the testing data.
    Args:
      test_x: Testing set inputs.
      test_y: Testing set expected outputs.
      batch_size: How big out batches are.
    Returns:
      Theano function for evaluating network accuracy. """
    index = TT.lscalar()
    outputs = TT.argmax(self._layer_stack, axis=1)

    batch_start = index * batch_size
    batch_end = (index + 1) * batch_size
    expected_outputs = test_y[batch_start:batch_end]
    accuracy = TT.mean(TT.eq(expected_outputs, outputs))

    tester = theano.function(inputs=[index], outputs=accuracy,
                             givens={self._inputs: \
                                     test_x[batch_start:batch_end]})
    return tester

  def _softmax_cross_entropy_with_logits(self, logits, labels):
    """ Return the cross-entropy of the prediction with the expected outputs.
    Args:
      logits: The actual outputs.
      labels: The expected outputs.
    Returns:
      Symbolic op for the cross-entropy opertaion.
    """
    softmax = TT.nnet.softmax(logits)
    argmax = TT.argmax(softmax, axis=1)
    cross = TT.nnet.categorical_crossentropy(softmax, labels)
    return cross

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

  @property
  def predict(self):
    """ Runs an actual prediction step for the network. It is intended that
    the result here get passed as the target of Session.run().
    Returns:
      The prediction operation. """
    return self._prediction_operation

  @property
  def train(self):
    """ Runs an SGD training step for the network. It is intended that the
    result here get passed as the target of Session.run().
    Returns:
      The training operation. """
    return self._optimizer

  @property
  def test(self):
    """ Runs a test on a single batch for the network, and returns the accuracy.
    Returns:
      The testing operation. """
    return self._tester
