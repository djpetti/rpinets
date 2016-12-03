""" Functions specific to neural networks. """


import logging

import numpy as np

from . import _store_backend as sb
from . import flow


sb.check_backend()

logger = logging.getLogger(__name__)


def dropout(layer_output, keep_prob, is_training, noise_shape=None, seed=None,
            name=None):
  """ Computes dropout for a layer.
  Args:
    layer_output: The activations to perform dropout on.
    keep_prob: The probability of keeping any specific element.
    is_training: A symbolic variable that keeps track if we are training or not.
                 If we are not training, dropout will be disabled.
    noise_shape: 1-D tensor representing the shape of generated keep/drop flags.
    seed: Seed for random number generation. This is only used in Tensorflow.
    name: The name of this operation. This is only used in Tensorflow.
  Returns:
    A tensor of the same shape as the input with dropout computed. """
  if sb.backend_name == "theano":
    if not noise_shape:
      # Just use the size of the input.
      noise_shape = layer_output.shape

    # Seed the random number generator if we need to.
    if seed:
      logger.debug("Seeding random number generator with %d" % (seed))
      sb.random.seed(seed)

    distribution = sb.random.binomial(size=noise_shape, p=keep_prob)
    dropped_out = sb.backend.tensor.switch(distribution, layer_output, 0)

  elif sb.backend_name == "tensorflow":
    dropped_out = sb.backend.nn.dropout(layer_output, keep_prob,
                                        noise_shape=noise_shape,
                                        seed=seed, name=name)

  # Don't do dropout if we're training.
  return flow.ifelse(is_training, dropped_out, layer_output * 0.5)

def relu(features, name=None):
  """ Computes the ReLU function.
  Args:
    features: The feature tensor to compute ReLU on.
    name: The name of this operation. This is only used in Tensorflow.
  Returns:
    The computed ReLU tensor. """
  if sb.backend_name == "theano":
    return sb.backend.tensor.nnet.relu(features)
  elif sb.backend_name == "tensorflow":
    return sb.backend.nn.relu(features, name=name)

def softmax(logits, name=None):
  """ Computes the softmax of a tensor.
  Args:
    logits: The tensor to compute the softmax of.
  Returns:
    The computed softmax. """
  if sb.backend_name == "theano":
    return sb.backend.tensor.nnet.softmax(logits)
  elif sb.backend_name == "tensorflow":
    return sb.backend.nn.softmax(logits, name=name)

def softmax_cross_entropy(logits, labels, name=None):
  """ Computes the cross-entropy of the prediction with the expected outputs.
  It also performs the softmax internally for scaling.
  Args:
    logits: The actual outputs.
    labels: The expected outputs.
  Returns:
    The computed cross-entropy. """
  if sb.backend_name == "theano":
    scaled = softmax(logits)
    return sb.backend.tensor.nnet.categorical_crossentropy(scaled, labels)

  elif sb.backend_name == "tensorflow":
    return sb.backend.nn.softmax_cross_entropy_with_logits(logits, labels,
                                                           name=name)

def initialize_xavier(weight_shape):
  """ An implementation of xavier initialization, based on code from here:
  https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py#L103-L177
  It works for convolutional and inner product layers. It assumes that all
  neurons are ReLU-activated, and draws from a normal distribution.
  Args:
    weigth_shape: The shape of the weights to initialize.
  Returns:
    A new array of weight values. """
  if len(weight_shape) == 4:
    # This is a convolutional layer.
    fan_out, fan_in, rows, cols = weight_shape
    receptive_field = rows * cols
  else:
    # This is an inner product layer.
    fan_out, fan_in = weight_shape
    receptive_field = 1

  # Compute the standard deviation.
  stddev = np.sqrt(2.0 / ((fan_out + fan_in) * receptive_field))

  # Get the weight array.
  weights = np.asarray(np.random.normal(0, stddev, size=weight_shape),
                       dtype="float32")
  return weights

def conv2d(inputs, filters, strides, border_mode):
  """ Performs a 2D convolution on the input.
  Args:
    inputs: The input to convolve.
    filters: The filters to use for the convolution.
    strides: Factor by which to subsample the output. This should be a tuple of
             the horizontal and vertical strides.
    border_mode: How to deal with the edge. Accepts "valid" and "half"
                 currently. The former only includes filter positions where
                 nothing goes off the edge, the latter pads the image with a
                 symmetric border.
  Returns:
    A convolution op. """
  if sb.backend_name == "theano":
    return sb.backend.tensor.nnet.conv2d(inputs, filters, subsample=strides,
                                         border_mode=border_mode)
  elif sb.backend_name == "tensorflow":
    # Tensorflow uses different names for its border modes.
    tf_border_modes = {"valid": "VALID", "half": "SAME"}
    border_mode = tf_border_modes[border_mode]

    # Tensorflow also wants the strides to be four elements long.
    strides = [1, strides[0], strides[1], 1]

    return sb.backend.nn.conv2d(inputs, filters, strides, border_mode,
                                data_format="NCHW")

def max_pool(inputs, kernel_shape, strides, padding):
  """ Performs a 2D max pooling operation on the input.
  Args:
    inputs: The input to max pool.
    kernel_shape: A tuple of the width and height of the kernel to use for
                  pooling.
    strides: The horizontal and vertical strides to use.
    border_mode: How to deal with the edge. Accepts "valid" and "half"
                 currently.
  Returns:
    A max pooling op. """
  if sb.backend_name == "theano":
    ignore_border = (padding == "VALID")
    return sb.backend.tensor.signal.pool.pool_2d(inputs, kernel_shape,
                                                 ignore_border=ignore_border,
                                                 st=strides)
  elif sb.backend_name == "tensorflow":
    return sb.backend.nn.max_pool(inputs, kernel_shape, strides, padding)

def local_response_normalization(data, depth_radius, bias, alpha, beta):
  """ Local response normalization, as described in the AlexNet paper.

  The 4D input tensor is interpreted as a 3D tensor of 1D vectors (Along the
  second dimension, as these are our feature maps that we want to normalize
  accross), and each vector is normalized independently.
  Args:
    data: The input data to use.
    depth_radius: Half-width of the 1D normalization window.
    bias: An offset.
    alpha: A scale factor.
    beta: An exponent. """
  if sb.backend_name == "theano":
    # Theano requires a sort of DIY implementation.
    half = depth_radius // 2
    # Square input data.
    square = math.square(data)

    batch, maps, x, y = data.shape
    extra_channels = sb.backend.tensor.alloc(0, batch, maps + 2 * half, x, y)
    square = sb.backend.tensor.set_subtensor( \
        extra_channels[:, half:half + maps, :, :], square)

    # Compute our scaling factor.
    scale = bias
    for i in range(depth_radius):
      scale += alpha * square[:, i:i + maps, :, :]
    scale = scale ** beta

    return data / scale

  elif sb.backend_name == "tensorflow":
    return sb.backend.nn.local_response_normalization(data, depth_radius,
                                                      bias=bias, alpha=alpha,
                                                      beta=beta)
