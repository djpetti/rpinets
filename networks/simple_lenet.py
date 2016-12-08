""" A very simple LeNet implementation. """


import numpy as np

from ..base_layer import *
from layers import ConvLayer, PoolLayer, NormalizationLayer
from simple_feedforward import FeedforwardNetwork


class LeNetClassifier(FeedforwardNetwork):
  """ A classifier built upon the Convolutional Neural Network as described by
  Yan LeCun. """

  def __init__(self, image_size, layers, outputs, train, test, batch_size,
               *args, **kwargs):
    """
    NOTE: For input images, this class will accept them in the format of a 4D
    tensor with the dimensions (batch_size, input_channels, input_rows,
    input_columns)
    Args:
      image_size: Size of the image. (width, height, channels)
      layers: A list of the layers to use for this network.
      outputs: The number of outputs of the network.
      train: The training dataset.
      test: The testing dataset.
      batch_size: The size of each image batch. """
    self._initialize_variables(layers, train, test, batch_size, *args, **kwargs)

    # We don't call the base class constructor here, because we want it to build
    # its network on top of our convolutional part.
    self.__build_model(image_size, outputs)

  def __split_layers(self, layers):
    """ Splits layers from the convolutional and feedforward parts of the
    network.
    Args:
      layers: The list of layers.
    Returns:
      The convolutional layers, and the feedforward ones."""
    conv = []
    feedforward = []
    for layer in layers:
      if (isinstance(layer, ConvLayer) \
          or isinstance(layer, PoolLayer) \
          or isinstance(layer, NormalizationLayer)):
        if feedforward:
          # The convolutional part of the network extends after the feedforward
          # part.
          raise ValueError("The feedforward layers must come after all \
                            convolutional ones.")
        # Convolutional layer.
        conv.append(layer)

      else:
        # Feedforward layer.
        feedforward.append(layer)

    return conv, feedforward

  def __initialize_weights(self, image_size, conv_layers, feedforward_inputs):
    """ Initializes tensors containing the weights and biases for each
    convolutional layer.
    Args:
      image_size: The size of the input image.
      conv_layers: A list of ConvLayer, PoolLayer and NormalizationLayer
      instances describing all the convolutional layers.
      feedforward_inputs: The number of inputs in the first feedforward layer. """
    image_x, image_y, channels = image_size

    self.__our_weights = []
    self.__our_biases = []
    # Keeps track of weight shapes.
    self.__weight_shapes = []
    # Extract only convolutional layers.
    only_convolution = []
    for layer in conv_layers:
      if isinstance(layer, ConvLayer):
        only_convolution.append(layer)

    input_feature_maps = channels

    for layer in only_convolution:
      # Initialize weights randomly.
      shape = [layer.feature_maps, input_feature_maps, layer.kernel_height,
               layer.kernel_width]
      self.__weight_shapes.append(shape)

      weights = self._make_initial_weights(shape, layer)
      self.__our_weights.append(weights)

      # Initialize biases.
      bias_values = np.full((layer.feature_maps,), layer.start_bias,
                             dtype="float32")
      bias = primitives.variable(bias_values)
      self.__our_biases.append(bias)

      input_feature_maps = layer.feature_maps

  def __add_layers(self, conv_layers, feedforward_layers, outputs):
    """ Adds as many convolutional layers to our model as there are elements in
    __weights.
    Args:
      conv_layers: A list containing specs for the convolution and maxpooling
      layers.
      feedforward_layers: A list denoting the number of inputs for each
      feedforward layer.
      outputs: The number of outputs of the network. """
    # Outputs from the previous layer that get used as inputs for the next
    # layer.
    next_inputs = self._inputs
    weight_index = 0
    for layer_spec in conv_layers:
      if isinstance(layer_spec, ConvLayer):
        # Convolution.
        weights = self.__our_weights[weight_index]
        output_feature_maps, _, _, _ = self.__weight_shapes[weight_index]

        conv = nnet.conv2d(next_inputs, weights,
                           (layer_spec.stride_width, layer_spec.stride_height),
                           layer_spec.border_mode)
        # Activation.
        bias = self.__our_biases[weight_index]
        next_inputs = nnet.relu(conv + bias.dimshuffle("x", 0, "x", "x"))

        weight_index += 1

      elif isinstance(layer_spec, NormalizationLayer):
        # Local normalization.
        next_inputs = nnet.local_response_normalization(next_inputs,
            layer_spec.depth_radius, layer_spec.bias, layer_spec.alpha,
            layer_spec.beta)

      else:
        # Max pooling.
        kernel_size = (layer_spec.kernel_width, layer_spec.kernel_height)
        stride_size = (layer_spec.stride_width, layer_spec.stride_height)
        next_inputs = nnet.max_pool(next_inputs, kernel_size,
                                    stride_size, "VALID")

      self._intermediate_activations.append(next_inputs)

    # Reshape convolution outputs so they can be used as inputs to the
    # feedforward network.
    flattened_inputs = math.flatten(next_inputs, 2)
    # Now that we're done building our weights, add them to the global list of
    # weights for gradient calculation.
    self._weights.extend(self.__our_weights)
    self._biases.extend(self.__our_biases)
    # Build the fully-connected part of the network.
    self._extend_with_feedforward(flattened_inputs, feedforward_layers, outputs)

  def __build_model(self, image_size, outputs):
    """ Constructs the graph for this model.
    Args:
      image_size: Size of the image.
      ouputs: The number of outputs of the network. """
    conv_layers, feedforward_layers = self.__split_layers(self._layers)

    # Initialize all the weights first.
    num_inputs = feedforward_layers[0].size
    self.__initialize_weights(image_size, conv_layers, num_inputs)

    # Create input and output variables.
    self._set_inputs_and_outptus()

    # Build actual layer model.
    self.__add_layers(conv_layers, feedforward_layers, outputs)
    # Now _layer_stack should contain the entire network.

    # Build cost function.
    self._cost = math.mean( \
        nnet.softmax_cross_entropy(self._layer_stack, self._expected_outputs))

    # Does an actual prediction.
    self._prediction_operation = self._build_predictor()
    # Evaluates the network's accuracy on the testing data.
    self._tester = self._build_tester()
