""" Utility functions that don't really belong in a specific class. """

import theano
import theano.tensor as TT


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
