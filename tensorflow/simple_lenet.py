""" A very simple LeNet implementation intended to be used for comparing
tensorflow to other libraries. """


import tensorflow as tf

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

  def __init__(self, image_size, conv_layers, feedforward_layers, outputs,
               batch_size=100):
    """
    Args:
      image_size: Size of the image. (width, height, channels)
      conv_layers: A list of convolutional layers, composed of ConvLayer
      instances.
      feedforward_layers: A list of ints denoting the number of inputs for each
      fully-connected layer.
      outputs: The number of outputs of the network.
      batch_size: The size of each image batch. """
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
      weights = tf.Variable(tf.random_normal(shape, stddev=1))
      self.__weights.append(weights)

    # The shapes of our convolution outputs will not be the same as those of our
    # inputs, which complicates things somewhat.
    image_x, image_y, channels = image_size
    output_shape = (image_x, image_y)
    # Calculate shape of output.
    for layer in conv_layers:
      out_shape_x = output_shape[0] - layer.kernel_width + 1
      out_shape_y = output_shape[1] - layer.kernel_height + 1
      # Factor in maxpooling.
      out_shape_x /= 2
      out_shape_y /= 2
      output_shape = (out_shape_x, out_shape_y)

    # Add last convolutional layer weights.
    final_x, final_y = output_shape
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
    next_inputs = self._inputs
    for weights in self.__weights:
      # Convolution.
      conv = tf.nn.conv2d(next_inputs, weights, strides=[1, 1, 1, 1],
                          padding="VALID")
      # Activation.
      num_outputs = weights.get_shape()[3]
      bias = tf.Variable(tf.constant(0.1, shape=[num_outputs]))
      conv = tf.nn.relu(conv + bias)
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
    self._inputs = tf.placeholder("float", [None, 28, 28, 1])
    self._expected_outputs = tf.placeholder("float", [None, outputs])

    # Build actual layer model.
    self.__add_layers(feedforward_layers, outputs)
    # Now _layer_stack should contain the entire network.

    # Build cost function.
    cost = tf.reduce_mean( \
        tf.nn.softmax_cross_entropy_with_logits(self._layer_stack,
                                                self._expected_outputs))

    # Learning rate decay.
    global_step = tf.Variable(0, trainable=False)
    start_learning_rate = 0.01
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                               1, 0.995)
    # RMS optimizer.
    self._optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9) \
        .minimize(cost, global_step=global_step)
    # Does an actual prediction.
    self._prediction_operation = tf.argmax(self._layer_stack, 1)
