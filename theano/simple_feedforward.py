""" A very simple FCFF NN intended to be used for comparing theano to other
libraries. """


from six.moves import cPickle as pickle
import sys

import theano
import theano.ifelse
import theano.tensor as TT

import numpy as np

import utils


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
    self._initialize_variables(train, test, batch_size)
    self.__build_model(layers, outputs)

  def __getstate__(self):
    """ Gets the state for pickling. """
    state = (self._weights, self._biases, self._layer_stack, self._cost,
             self._inputs, self._expected_outputs, self.__trainer_type,
             self.__train_params, self._global_step, self._srng,
             self._training, self._used_training, self._intermediate_activations)
    return state

  def __setstate__(self, state):
    """ Sets the state for unpickling. """
    self._weights, self._biases, self._layer_stack, self._cost, self._inputs, \
    self._expected_outputs, self.__trainer_type, self.__train_params, \
    self._global_step, self._srng, self._training, \
    self._used_training, self._intermediate_activations = state

  def _initialize_variables(self, train, test, batch_size):
    """ Initializes variables that are common to all subclasses.
    Args:
      train: Training set.
      test: Testing set.
      batch_size: Size of each minibatch. """
    self._train_x, self._train_y = train
    self._test_x, self._test_y = test
    self._batch_size = batch_size

    # These are the weights and biases that will be used for calculating
    # gradients.
    self._weights = []
    self._biases = []

    self._optimizer = None
    self.__trainer_type = None
    self.__train_params = ()
    # A global step to use for learning rate decays.
    self._global_step = theano.shared(0)

    self._srng = TT.shared_randomstreams.RandomStreams()
    self._training = TT.lscalar("training")
    # Keeps track of whether _training actually gets used, because Theano is
    # annoying about initializing unused variables.
    self._used_training = False

    self._intermediate_activations = []

  def _make_initial_weights(self, weight_shape, layer):
    """ A helper function that generates random starting values for the weights.
    Args:
      weight_shape: The shape of the weights tensor.
      layer: The layer the weights are for.
    Returns:
      A shared variable containing the weights. """
    if layer.weight_init == "xavier":
      # Use xavier initialization.
      weight_values = utils.initialize_xavier(weight_shape)
    elif layer.weight_init == "gaussian":
      # Use gaussian initialization.
      dist = np.random.normal(layer.weight_mean, layer.weight_stddev,
                              size=weight_shape)
      weight_values = np.asarray(dist, dtype=theano.config.floatX)

    return theano.shared(weight_values)

  def __initialize_weights(self, layers, outputs):
    """ Initializes tensors containing the weights and biases for each layer.
    Args:
      layers: A list denoting the number of inputs of each layer.
      outputs: The number of outputs of the network. """
    self.__our_weights = []
    self.__our_biases = []
    # Keeps track of weight shapes because Theano is annoying about that.
    self.__weight_shapes = []
    # This is in case we have a single hidden layer.
    fan_out = layers[0].size
    for i in range(0, len(layers) - 1):
      fan_in = layers[i].size
      fan_out = layers[i + 1].size

      # Initialize weights randomly.
      weights = self._make_initial_weights((fan_in, fan_out), layers[i])
      self.__our_weights.append(weights)
      self.__weight_shapes.append((fan_in, fan_out))

      # Initialize biases.
      bias_values = np.full((fan_out,), layers[i].start_bias,
                            dtype=theano.config.floatX)
      bias = theano.shared(bias_values)
      self.__our_biases.append(bias)

    # Include outputs also.
    weights = self._make_initial_weights((fan_out, outputs), layers[i])
    self.__our_weights.append(weights)
    bias_values = np.full((outputs,), layers[i].start_bias,
                          dtype=theano.config.floatX)
    self.__our_biases.append(theano.shared(bias_values))

    self.__weight_shapes.append((fan_out, outputs))

  def __add_layers(self, first_inputs, layers):
    """ Adds as many hidden layers to our model as there are elements in
    __weights.
    Args:
      first_inputs: The tensor to use as inputs to the first hidden layer.
      layers: The list of all the layers in this network. """
    # Outputs from the previous layer that get used as inputs for the next
    # layer.
    next_inputs = first_inputs
    for i in range(0, len(self.__our_weights)):
      weights = self.__our_weights[i]
      _, fan_out = self.__weight_shapes[i]
      layer = layers[i]
      bias = self.__our_biases[i]

      sums = TT.dot(next_inputs, weights) + bias
      if i < len(self.__our_weights) - 1:
        next_inputs = TT.nnet.relu(sums)
      else:
        # For the last layer, we don't use an activation function.
        next_inputs = sums

      if layer.dropout:
        # Do dropout.
        self._used_training = True
        dropped_out = TT.switch(self._srng.binomial(size=next_inputs.shape,
                                                    p=0.5), next_inputs, 0)
        next_inputs = theano.ifelse.ifelse(self._training, dropped_out,
                                           next_inputs * 0.5)

      self._intermediate_activations.append(next_inputs)

    self._layer_stack = next_inputs
    # Now that we're done building our weights, add them to the global list of
    # weights for gradient calculation.
    self._weights.extend(self.__our_weights)
    self._biases.extend(self.__our_biases)

  def __build_model(self, layers, outputs):
    """ Actually constructs the graph for this model.
    Args:
      layers: A list denoting the number of inputs of each layer.
      outputs: The number of outputs of the network. """
    # Initialize all the weights first.
    self.__initialize_weights(layers, outputs)

    # Inputs and outputs.
    self._inputs = TT.fmatrix("inputs")
    self._expected_outputs = TT.ivector("expected_outputs")

    # Build actual layer model.
    self.__add_layers(self._inputs, layers)

    # Cost function.
    self._cost = TT.mean( \
        self._softmax_cross_entropy_with_logits(self._layer_stack,
                                                self._expected_outputs))

    # Does an actual prediction.
    self._prediction_operation = self._build_predictor(self._test_x,
                                                       self._batch_size)
    # Evaluates the network's accuracy on the testing data.
    self._tester = self._build_tester(self._test_x, self._test_y,
                                      self._batch_size)

  def __make_givens(self, batch_x, batch_y, index, batch_size, training):
    """ Makes the givens dictionary for a training or testing function.
    Args:
      batch_x: The input data.
      batch_y: The input labels.
      index: The index into the batch where we are starting.
      batch_size: The size of the batch.
      training: Whether we are training or not.
    Returns:
      The dictionary to use for givens. """
    batch_start = index * batch_size
    batch_end = (index + 1) * batch_size

    givens = {self._inputs: batch_x[batch_start:batch_end],
              self._expected_outputs: batch_y[batch_start:batch_end]}

    if self._used_training:
      givens[self._training] = training

    return givens

  def _build_sgd_trainer(self, cost, learning_rate, momentum, weight_decay,
                         train_x, train_y, batch_size):
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
    params = self._weights + self._biases

    # Tell it how to update the parameters.
    updates, grads = utils.momentum_sgd(cost, params, learning_rate, momentum,
                                        weight_decay)
    # Update the global step too.
    updates.append((self._global_step, self._global_step + 1))

    # Index to a minibatch.
    index = TT.lscalar()
    # Create the actual function.
    givens = self.__make_givens(train_x, train_y, index, batch_size, 1)
    trainer = theano.function(inputs=[index], outputs=[cost, learning_rate,
                                                       self._global_step],
                              updates=updates,
                              givens=givens)
    return trainer

  def _build_rmsprop_trainer(self, cost, learning_rate, rho, epsilon, train_x,
                             train_y, batch_size):
    """ Builds a new RMSProp trainer.
    Args:
      cost: The cost function we are using.
      learning_rate: The learning rate to use for training.
      rho: Weight decay.
      epsilon: Shift factor for gradient scaling.
      train_x: Training set inputs.
      train_y: Training set expected outputs.
      batch_size: How big our batches are.
    Returns:
      Theano function for training the network. """
    params = self._weights + self._biases

    updates = utils.rmsprop(cost, params, learning_rate, rho, epsilon)
    # Update the global step too.
    updates.append((self._global_step, self._global_step + 1))

    # Index to a minibatch.
    index = TT.lscalar()
    # Create the actual function.
    givens = self.__make_givens(train_x, train_y, index, batch_size, 1)
    trainer = theano.function(inputs=[index], outputs=[cost, learning_rate,
                                                       self._global_step],
                              updates=updates,
                              givens=givens)
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
    givens={self._inputs: test_x[batch_start:batch_end]}
    if self._used_training:
      givens[self._training] = 0

    predictor = theano.function(inputs=[index], outputs=outputs,
                                givens=givens)
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
    softmax = TT.nnet.softmax(self._layer_stack)
    argmax = TT.argmax(softmax, axis=1)

    batch_start = index * batch_size
    batch_end = (index + 1) * batch_size
    expected_outputs = test_y[batch_start:batch_end]
    accuracy = TT.mean(TT.eq(expected_outputs, argmax))
    givens={self._inputs: test_x[batch_start:batch_end]}
    if self._used_training:
      givens[self._training] = 0

    tester = theano.function(inputs=[index], outputs=accuracy,
                             givens=givens)
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
    self.__add_layers(inputs, layers)

  def use_sgd_trainer(self, learning_rate, momentum=0.9, weight_decay=0.0005,
                      decay_rate=1, decay_steps=0):
    """ Tells it to use SGD to train the network.
    Args:
      learning_rate: The learning rate to use for training.
      decay_rate: An optional exponential decay rate for the learning rate.
      decay_steps: An optinal number of steps to decay in. """
    # Save these for use when saving and loading the network.
    self.__trainer_type = "sgd"
    self.__train_params = (learning_rate, momentum, weight_decay,
                           decay_rate, decay_steps)

    # Handle learning rate decay.
    decayed_learning_rate = \
        utils.exponential_decay(learning_rate, self._global_step, decay_steps,
                                decay_rate)

    self._optimizer = self._build_sgd_trainer(self._cost, decayed_learning_rate,
                                              momentum, weight_decay,
                                              self._train_x, self._train_y,
                                              self._batch_size)

  def use_rmsprop_trainer(self, learning_rate, rho, epsilon, decay_rate=1,
                          decay_steps=0):
    """ Tells it to use RMSProp to train the network.
    Args:
      learning_rate: The learning rate to use for training.
      rho: Weight decay.
      epsilon: Shift factor for gradient scaling.
      decay_rate: An optinal exponential decay rate for the learning rate.
      decay_steps: An optional number of steps to decay in. """
    # Save these for use when saving and loading the network.
    self.__trainer_type = "rmsprop"
    self.__train_params = (learning_rate, rho, epsilon, decay_rate, decay_steps)

    # Handle learning rate decay.
    decayed_learning_rate = \
        utils.exponential_decay(learning_rate, self._global_step, decay_steps,
                                decay_rate)

    self._optimizer = self._build_rmsprop_trainer(self._cost, decayed_learning_rate,
                                                  rho, epsilon, self._train_x,
                                                  self._train_y,
                                                  self._batch_size)

  @property
  def predict(self):
    """ Runs an actual prediction step for the network.
    Returns:
      The prediction operation. """
    return self._prediction_operation

  @property
  def train(self):
    """ Runs an training step for the network.
    Returns:
      The training operation. """
    if not self._optimizer:
      raise RuntimeError("No trainer is configured!")

    return self._optimizer

  @property
  def test(self):
    """ Runs a test on a single batch for the network, and returns the accuracy.
    Returns:
      The testing operation. """
    return self._tester

  def save(self, filename):
    """ Saves the network to a file.
    Args:
      filename: The name of the file to save to. """
    file_object = open(filename, "wb")
    pickle.dump(self, file_object, protocol=pickle.HIGHEST_PROTOCOL)
    file_object.close()

  @classmethod
  def load(cls, filename, train, test, batch_size, learning_rate=None):
    """ Loads the network from a file.
    Args:
      filename: The name of the file to load from.
      train: The training dataset to use. If this is None, it won't build a
      training function.
      test: The testing dataset to use. If this is None, it won't build
      predictor and tester functions.
      batch_size: The batch size to use.
      learning_rate: Allows us to specify a new learning rate for the network.
    Returns:
      The loaded network. """
    file_object = open(filename, "rb")
    network = pickle.load(file_object)
    file_object.close()

    network._batch_size = batch_size

    # If they didn't give us a test dataset, just don't build these functions.
    if test:
      # Set datasets.
      network._test_x, network._test_y = test

      # Build predictor.
      network._prediction_operation = \
          network._build_predictor(network._test_x, network._batch_size)
      # Build tester.
      network._tester = network._build_tester(network._test_x, network._test_y,
                                              network._batch_size)

    if train:
      # Set datasets.
      network._train_x, network._train_y = train

      # Reconstruct the specified trainer.
      builder = None
      if network.__trainer_type == "rmsprop":
        builder = network.use_rmsprop_trainer
      if network.__trainer_type == "sgd":
        builder = network.use_sgd_trainer

      if builder:
        params = list(network.__train_params)
        if learning_rate != None:
          # Use custom learning rate.
          params[0] = learning_rate
        builder(*params)

    return network

  def get_train_x(self):
    """ Returns: The input training image buffer. """
    return self._train_x

  def get_test_x(self):
    """ Returns: The input testing image buffer. """
    return self._test_x
