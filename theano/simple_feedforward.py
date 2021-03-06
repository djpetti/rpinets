""" A very simple FCFF NN intended to be used for comparing theano to other
libraries. """


from six.moves import cPickle as pickle
import logging
import sys

import theano
import theano.ifelse
import theano.tensor as TT

import numpy as np

import learning_rates
import utils


logger = logging.getLogger(__name__)


class FeedforwardNetwork(object):
  """ A simple, fully-connected feedforward neural network. """

  def __init__(self, layers, outputs, train, test, batch_size,
               patch_separation=None):
    """
    Args:
      layers: A list of layer structures for the layers in the network.
      outputs: The number of outputs of the network.
      train: Training dataset, should be pair of inputs and expected outputs.
      test: Testing dataset, should be pair of inputs and expected outputs.
      batch_size: The size of each minibatch.
      patch_separation: Specifies the distance between each patch of a given
                        image. If not provided, it is assumed to be the same
                        as the batch size. """
    self._initialize_variables(layers, train, test, batch_size,
                               patch_separation)
    self.__build_model(outputs)

  def __getstate__(self):
    """ Gets the state for pickling. """
    state = (self._weights, self._biases, self._layer_stack, self._cost,
             self._inputs, self._expected_outputs, self.__trainer_type,
             self.__train_params, self._global_step, self._srng,
             self._training, self._used_training,
             self._intermediate_activations,
             self._patch_separation, self._layers, self.__train_kw_params)
    return state

  def __setstate__(self, state):
    """ Sets the state for unpickling. """
    self._weights, self._biases, self._layer_stack, self._cost, self._inputs, \
    self._expected_outputs, self.__trainer_type, self.__train_params, \
    self._global_step, self._srng, self._training, \
    self._used_training, self._intermediate_activations, \
    self._patch_separation, self._layers, self.__train_kw_params = state

  def _initialize_variables(self, layers, train, test, batch_size,
                            patch_separation=None):
    """ Initializes variables that are common to all subclasses.
    Args:
      layers: The layers of the network.
      train: Training set.
      test: Testing set.
      batch_size: Size of each minibatch.
      patch_separation: Distance between each patch of a given image. """
    self._layers = layers

    self._train_x, self._train_y = train
    self._test_x, self._test_y = test
    self._batch_size = batch_size

    if patch_separation:
      self._patch_separation = patch_separation
    else:
      self._patch_separation = self._batch_size

    # These are the weights and biases that will be used for calculating
    # gradients.
    self._weights = []
    self._biases = []

    self._optimizer = None
    self.__trainer_type = None
    self.__train_params = ()
    self.__train_kw_params = {}
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
      layers: A list of layer structures for the layers to initialize.
      outputs: The number of outputs of the network.
      fan_in: Overrides the default fan-in for the layers with a custom one. """
    self.__our_weights = []
    self.__our_biases = []
    # Keeps track of weight shapes because Theano is annoying about that.
    self.__weight_shapes = []

    fan_out = layers[0].size
    i = 0
    for i in range(0, len(layers) - 1):
      fan_in = layers[i].size
      fan_out = layers[i + 1].size

      # Initialize weights randomly.
      weights = self._make_initial_weights((fan_in, fan_out), layers[i])
      self.__our_weights.append(weights)
      logger.debug("Adding weights with shape (%d, %d)." % (fan_in, fan_out))
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

  def replace_bottom_layers(self, replace_with, outputs):
    """ Replaces a set of layers at the bottom of the network.
    NOTE: Currently, this only works for layers that were added in the
    feedforward part of the network.
    Args:
      replace_with: Set of replacement layers.
      outputs: The new number of outputs the network will have. """
    #TODO (danielp): Make this work for convolutional layers.
    removing_layers = len(replace_with) + 1
    logger.info("Replacing %d layers at bottom of network..." % \
                (removing_layers))

    # First, chop off the data for the layers we're getting rid of.
    self._weights = self._weights[:-removing_layers]
    self._biases = self._biases[:-removing_layers]
    self._intermediate_activations = \
        self._intermediate_activations[:-removing_layers]
    # self._layers doesn't contain the outputs.
    if replace_with:
      self._layers = self._layers[:-len(replace_with)]

    # We're going to also have to redo the layer above, so it can interface with
    # the new ones.
    interface_layer_specs = self._layers[-1]
    replace_with.insert(0, interface_layer_specs)
    logger.debug("Have %d inputs from the old stack." % \
                 (interface_layer_specs.size))

    # Build weights for new layers.
    self.__initialize_weights(replace_with, outputs)

    # When we build layers, we need something to build them off of, which
    # ideally should be the last layer that we didn't touch.
    last_unchanged_layer = self._intermediate_activations[-1]
    start_index = len(self.__our_weights) - removing_layers
    # Now, actually build the new layers.
    self.__add_layers(last_unchanged_layer, replace_with)

    # Update the list of layers.
    self._layers.extend(replace_with[1:])

    # Since the layer stack changed, we have to update the cost.
    self._cost = TT.mean( \
        self._softmax_cross_entropy_with_logits(self._layer_stack,
                                                self._expected_outputs))

    # Now we changed everything, so we have to rebuild all our functions.
    logger.debug("Rebuilding functions...")
    train = (self._train_x, self._train_y)
    test = (self._test_x, self._test_y)
    self.__rebuild_functions(train, test)

  def __build_model(self, outputs):
    """ Actually constructs the graph for this model.
    Args:
      outputs: The number of outputs of the network. """
    # Initialize all the weights first.
    self.__initialize_weights(self._layers, outputs)

    # Inputs and outputs.
    self._inputs = TT.fmatrix("inputs")
    self._expected_outputs = TT.ivector("expected_outputs")

    # Build actual layer model.
    self.__add_layers(self._inputs, self._layers)

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

  def __rebuild_functions(self, train, test, learning_rate=None,
                          train_layers=None):
    """ Reconstruct functions from a saved network.
    Args:
      train: Train dataset, with data and labels.
      test: Test dataset, with data and labels.
      learning_rate: Allows the user to override the saved learning rate.
      train_layers: List of layers to actually train. All the rest will be left
                    untouched. """
    # If they didn't give us a test dataset, just don't build these functions.
    if test:
      # Set datasets.
      self._test_x, self._test_y = test

      # Build predictor.
      self._prediction_operation = \
          self._build_predictor(self._test_x, self._batch_size)
      # Build tester.
      self._tester = self._build_tester(self._test_x, self._test_y,
                                        self._batch_size)

    if train:
      # Set datasets.
      self._train_x, self._train_y = train

      # Reconstruct the specified trainer.
      builder = None
      if self.__trainer_type == "rmsprop":
        builder = self.use_rmsprop_trainer
      if self.__trainer_type == "sgd":
        builder = self.use_sgd_trainer

      if builder:
        params = list(self.__train_params)
        kw_params = self.__train_kw_params

        if learning_rate != None:
          # Use custom learning rate.
          params[0] = learning_rate
        if train_layers != None:
          # Use custom list of layers to train.
          kw_params["train_layers"] = train_layers
        builder(*params, **kw_params)


  def __make_params(self, train_layers):
    """ Creates the list of parameters to update on each training step.
    Args:
      train_layers: A list of the layers to actually train. If this is None, all
      of them will be used.
    Returns:
      A list of the parameters to update. """
    params = []
    if train_layers == None:
      params = self._weights + self._biases
    else:
      for i in train_layers:
        params.append(self._weights[i])
      for i in train_layers:
        params.append(self._biases[i])

    return params

  def _build_sgd_trainer(self, cost, learning_rate, momentum, weight_decay,
                         train_x, train_y, batch_size, train_layers=None):
    """ Builds a new SGD trainer for the network.
    Args:
      cost: The cost function we are using.
      learning_rate: The learning rate to use for training. Should be a class
                     from learning_rates.
      train_x: Training set inputs.
      train_y: Training set expected outputs.
      batch_size: How big our batches are.
      train_layers: If specified, this should be a list of the layers that are
      actually going to be trained. Otherwise, all of them will be trained.
    Returns:
      Theano function for training the network. """
    # Compute gradients for all parameters.
    params = self.__make_params(train_layers)

    # Tell it how to update the parameters.
    use_lr = learning_rate.get(self._global_step)
    updates, grads = utils.momentum_sgd(cost, params, use_lr, momentum,
                                        weight_decay)
    # Update the global step too.
    updates.append((self._global_step, self._global_step + 1))

    # Index to a minibatch.
    index = TT.lscalar()
    # Create the actual function.
    givens = self.__make_givens(train_x, train_y, index, batch_size, 1)
    trainer = theano.function(inputs=[index], outputs=[cost, use_lr,
                                                       self._global_step],
                              updates=updates,
                              givens=givens)
    return trainer

  def _build_rmsprop_trainer(self, cost, learning_rate, rho, epsilon, train_x,
                             train_y, batch_size, train_layers=None):
    """ Builds a new RMSProp trainer.
    Args:
      cost: The cost function we are using.
      learning_rate: The learning rate to use for training. Should be a class
                     from learning_rates.
      rho: Weight decay.
      epsilon: Shift factor for gradient scaling.
      train_x: Training set inputs.
      train_y: Training set expected outputs.
      batch_size: How big our batches are.
      train_layers: If specified, this should be a list of the layers that are
      actually going to be trained. Otherwise, all of them will be trained.
    Returns:
      Theano function for training the network. """
    params = self.__make_params(train_layers)

    use_lr = learning_rate.get(self._global_step)
    updates = utils.rmsprop(cost, params, use_lr, rho, epsilon)
    # Update the global step too.
    updates.append((self._global_step, self._global_step + 1))

    # Index to a minibatch.
    index = TT.lscalar()
    # Create the actual function.
    givens = self.__make_givens(train_x, train_y, index, batch_size, 1)
    trainer = theano.function(inputs=[index], outputs=[cost, use_lr,
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
    softmax = TT.nnet.softmax(self._layer_stack)
    outputs = TT.argmax(softmax, axis=1)

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
                      train_layers=None):
    """ Tells it to use SGD to train the network.
    Args:
      learning_rate: The learning rate to use for training.
      momentum: The momentum to use for SGD.
      weight_decay: The weight decay to use for SGD.
      train_layers: A list of the layer indices to actually train. If not
      specified, it will train all of them. """
    # Save these for use when saving and loading the network.
    self.__trainer_type = "sgd"
    self.__train_params = (learning_rate, momentum, weight_decay)
    self.__train_kw_params = {"train_layers": train_layers}

    # If the user simply passes in a number for the learning rate, we'll just
    # assume it's fixed.
    if not isinstance(learning_rate, learning_rates._LearningRate):
      logger.debug("Assuming fixed learning rate: %f" % (learning_rate))
      learning_rate = learning_rates.Fixed(learning_rate)

    self._optimizer = self._build_sgd_trainer(self._cost, learning_rate,
                                              momentum, weight_decay,
                                              self._train_x, self._train_y,
                                              self._batch_size, train_layers)

  def use_rmsprop_trainer(self, learning_rate, rho, epsilon, train_layers=None):
    """ Tells it to use RMSProp to train the network.
    Args:
      learning_rate: The learning rate to use for training.
      rho: Weight decay.
      epsilon: Shift factor for gradient scaling.
      train_layers: A list of the layer indices to actually train. If not
      specified, it will train all of them. """
    # Save these for use when saving and loading the network.
    self.__trainer_type = "rmsprop"
    self.__train_params = (learning_rate, rho, epsilon)
    self.__train_kw_params = {"train_layers": train_layers}

    # If the user simply passes in a number for the learning rate, we'll just
    # assume it's fixed.
    if not isinstance(learning_rate, learning_rates._LearningRate):
      logger.debug("Assuming fixed learning rate: %f" % (learning_rate))
      learning_rate = learning_rates.Fixed(learning_rate)

    self._optimizer = self._build_rmsprop_trainer(self._cost, decayed_learning_rate,
                                                  rho, epsilon, self._train_x,
                                                  self._train_y,
                                                  self._batch_size,
                                                  train_layers)

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
  def load(cls, filename, train, test, batch_size, learning_rate=None,
           train_layers=None):
    """ Loads the network from a file.
    Args:
      filename: The name of the file to load from.
      train: The training dataset to use. If this is None, it won't build a
      training function.
      test: The testing dataset to use. If this is None, it won't build
      predictor and tester functions.
      batch_size: The batch size to use.
      learning_rate: Allows us to specify a new learning rate for the network.
      train_layers: List of layers to actually train, the rest will be left
                    untouched.
    Returns:
      The loaded network. """
    file_object = open(filename, "rb")
    network = pickle.load(file_object)
    file_object.close()

    network._batch_size = batch_size

    network.__rebuild_functions(train, test, learning_rate=learning_rate,
                                train_layers=train_layers)

    return network

  def get_train_x(self):
    """ Returns: The input training image buffer. """
    return self._train_x

  def get_test_x(self):
    """ Returns: The input testing image buffer. """
    return self._test_x

  def reset_global_step(self, value=0):
    """ Resets the global step variable.
    Args:
      value: The value to reset it to. This defaults to zero. """
    logger.info("Resetting global step value to %d." % (value))
    self._global_step.set_value(value)
