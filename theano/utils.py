""" Utility functions that don't really belong in a specific class. """

import theano
import theano.tensor as TT

import numpy as np


def rmsprop(cost, params, lr, rho, epsilon):
  """ Performs RMS Prop on a set of parameters. This code is sourced from here:
  https://github.com/Newmu/Theano-Tutorials/blob/master/5_convolutional_net.py
  Args:
    cost: The symbolic cost function.
    params: The parameters that we want to update.
    lr: Learning rate.
    rho: Weight decay.
    epsilon: Shift factor for gradient scaling.
  Returns:
    List of symbolic updates for the parameters. """
  grads = TT.grad(cost=cost, wrt=params)
  updates = []
  for p, g in zip(params, grads):
    acc = theano.shared(p.get_value() * 0.)
    acc_new = rho * acc + (1 - rho) * g ** 2
    gradient_scaling = TT.sqrt(acc_new + epsilon)
    g = g / gradient_scaling
    updates.append((acc, acc_new))
    updates.append((p, p - lr * g))
  return updates

def momentum_sgd(cost, params, lr, momentum, weight_decay):
  """ Performs SGD, as described in the AlexNet paper.
  Args:
    cost: The symbolic cost function.
    params: The parameters that we want to update.
    lr: The learning rate.
    momentum: Momentum to use.
    weight_decay: Weight decay to use.
  Returns:
    List of symbolic updates for the parameters. """
  grads = TT.grad(cost=cost, wrt=params)
  updates = []
  for p, g in zip(params, grads):
    v = theano.shared(p.get_value() * 0.)
    v_next = momentum * v - weight_decay * lr * p - lr * g
    updates.append((v, v_next))

    updates.append((p, p + v_next))

  return updates, g

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
  half = depth_radius // 2
  # Square input data.
  square = TT.sqr(data)

  batch, maps, x, y = data.shape
  extra_channels = TT.alloc(0, batch, maps + 2 * half, x, y)
  square = TT.set_subtensor(extra_channels[:, half:half + maps, :, :], square)

  # Compute our scaling factor.
  scale = bias
  for i in range(depth_radius):
    scale += alpha * square[:, i:i + maps, :, :]
  scale = scale ** beta

  return data / scale

def exponential_decay(learning_rate, global_step, decay_steps, decay_rate):
  """ Applies an exponential decay to the learning rate.
  Args:
    learning_rate: The initial learning rate.
    global_step: Global step to use for the decay computation.
    decay_steps: How many steps it takes to reach our full decay rate.
    decay_rate: The decay rate.
  Returns:
    An exponentially decayed learning rate. """
  rate = learning_rate * decay_rate ** (global_step / decay_steps)
  return TT.cast(rate, theano.config.floatX)

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
                       dtype=theano.config.floatX)
  return weights
