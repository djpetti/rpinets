""" A very simple LeNet implementation intended to be used for comparing
theano to other libraries. """


import theano
import theano.tensor as TT
import theano.tensor.signal.pool as pool

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
      # Stride size for convolution. (Defaults to (1, 1))
      self.stride_width = kwargs.get("stride_width", 1)
      self.stride_height = kwargs.get("stride_height", 1)

  class PoolLayer(object):
    """ A simple class to handle the specification of maxpooling layers. """

    def __init__(self, *args, **kwargs):
      # Maxpooling kernel size. (Defaults to 2x2.)
      self.kernel_width = kwargs.get("kernel_width", 2)
      self.kernel_height = kwargs.get("kernel_height", 2)
      # Maxpooling stride size. (Defaults to the same as the kernel.)
      self.stride_width = kwargs.get("stride_width", self.kernel_width)
      self.stride_height = kwargs.get("stride_height", self.kernel_height)

  def __init__(self, image_size, conv_layers, feedforward_layers, outputs,
               train, test, batch_size):
    """
    NOTE: For input images, this class will accept them in the format of a 4D
    tensor with the dimensions (batch_size, input_channels, input_rows,
    input_columns)
    Args:
      image_size: Size of the image. (width, height, channels)
      conv_layers: A list of convolutional layers and maxpooling layers,
      composed of ConvLayer and PoolLayer instances.
      feedforward_layers: A list of ints denoting the number of inputs for each
      fully-connected layer.
      outputs: The number of outputs of the network.
      train: The training dataset.
      test: The testing dataset.
      batch_size: The size of each image batch. """
    self._train_x, self._train_y = train
    self._test_x, self._test_y = test
    self._batch_size = batch_size

    # These are the weights and biases that will be used for calculating
    # gradients.
    self._weights = []
    self._biases = []

    self._print_op = theano.printing.Print("Debug: ")

    # We don't call the base class constructor here, because we want it to build
    # its network on top of our convolutional part.
    self.__build_model(image_size, conv_layers, feedforward_layers, outputs)

  def __initialize_weights(self, image_size, conv_layers, feedforward_inputs):
    """ Initializes tensors containing the weights for each convolutional layer.
    Args:
      image_size: The size of the input image.
      conv_layers: A list of ConvLayer and PoolLayer instances describing all
      the convolutional layers.
      feedforward_inputs: The number of inputs in the first feedforward layer. """
    image_x, image_y, channels = image_size

    self.__our_weights = []
    # Keeps track of weight shapes because Theano is annoying about that.
    self.__weight_shapes = []
    # Extract only convolutional layers.
    only_convolution = []
    for layer in conv_layers:
      if isinstance(layer, self.ConvLayer):
        only_convolution.append(layer)
    for i in range(0, len(only_convolution) - 1):
      first_layer = only_convolution[i]
      next_layer = only_convolution[i + 1]

      # Initialize weights randomly.
      shape = [next_layer.feature_maps, first_layer.feature_maps,
               first_layer.kernel_height, first_layer.kernel_width]
      self.__weight_shapes.append(shape)

      weights_values = np.asarray(np.random.normal(size=shape),
                                  dtype=theano.config.floatX)
      weights = theano.shared(weights_values)
      self.__our_weights.append(weights)

    # The shapes of our convolution outputs will not be the same as those of our
    # inputs, which complicates things somewhat.
    output_shape = (image_x, image_y)
    # Calculate shape of output.
    for layer in conv_layers:
      if isinstance(layer, self.ConvLayer):
        out_shape_x = output_shape[0] - layer.kernel_width + 1
        out_shape_y = output_shape[1] - layer.kernel_height + 1
      else:
        # Factor in maxpooling.
        out_shape_x /= 2
        out_shape_y /= 2
      output_shape = (out_shape_x, out_shape_y)

    # Add last convolutional layer weights.
    final_x, final_y = output_shape
    shape = [feedforward_inputs / final_x / final_y / channels,
             next_layer.feature_maps,
             next_layer.kernel_height,
             next_layer.kernel_width]
    self.__weight_shapes.append(shape)

    weights_values = np.asarray(np.random.normal(size=shape),
                                dtype=theano.config.floatX)
    weights = theano.shared(weights_values)
    self._pweights = self._print_op(weights)
    self.__our_weights.append(weights)

  def __add_layers(self, conv_layers, feedforward_layers, outputs):
    """ Adds as many convolutional layers to our model as there are elements in
    __weights.
    Args:
      conv_layers: A list containing specs for the convolution and maxpooling
      layers.
      feedforward_layers: A list denoting the number of inputs for each
      feedforward layer.
      outputs: The number of outputs of the network. """
    our_biases = []
    # Outputs from the previous layer that get used as inputs for the next
    # layer.
    next_inputs = self._inputs
    weight_index = 0
    for layer_spec in conv_layers:
      if isinstance(layer_spec, self.ConvLayer):
        # Convolution.
        weights = self.__our_weights[weight_index]
        output_feature_maps, _, _, _ = self.__weight_shapes[weight_index]
        weight_index += 1

        conv = TT.nnet.conv2d(next_inputs, weights,
                              subsample=(layer_spec.stride_width,
                                         layer_spec.stride_height),
                              border_mode="valid")
        # Activation.
        bias_values = np.zeros((output_feature_maps,), dtype=theano.config.floatX)
        bias = theano.shared(bias_values)
        our_biases.append(bias)
        next_inputs = TT.nnet.relu(conv + bias.dimshuffle("x", 0, "x", "x"))
      else:
        # Max pooling.
        kernel_size = (layer_spec.kernel_width, layer_spec.kernel_height)
        stride_size = (layer_spec.stride_width, layer_spec.stride_height)
        next_inputs = pool.pool_2d(next_inputs, kernel_size,
                                  ignore_border=True,
                                  st=stride_size)

    # Reshape convolution outputs so they can be used as inputs to the
    # feedforward network.
    num_inputs = feedforward_layers[0]
    flattened_inputs = TT.reshape(next_inputs, [self._batch_size, num_inputs])
    self._pflat = self._print_op(flattened_inputs)
    # Now that we're done building our weights, add them to the global list of
    # weights for gradient calculation.
    self._weights.extend(self.__our_weights)
    self._biases.extend(our_biases)
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
    self._inputs = TT.ftensor4("inputs")
    self._expected_outputs = TT.ivector("expected_outputs")

    # Build actual layer model.
    self.__add_layers(conv_layers, feedforward_layers, outputs)
    # Now _layer_stack should contain the entire network.

    # Build cost function.
    cost = TT.mean( \
        self._softmax_cross_entropy_with_logits(self._layer_stack,
                                                self._expected_outputs))

    # RMSProp optimizer.
    self._optimizer = self._build_rmsprop_trainer(cost, 0.001, 0.9, 1e-6,
                                                  self._train_x, self._train_y,
                                                  self._batch_size)
    # Does an actual prediction.
    self._prediction_operation = self._build_predictor(self._test_x,
                                                       self._batch_size)
    # Evaluates the network's accuracy on the testing data.
    self._tester = self._build_tester(self._test_x, self._test_y,
                                      self._batch_size)
