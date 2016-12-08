""" Utility functions that don't really belong in a specific class. """

import theano.tensor as TT

import numpy as np

from ..base_layer import primitives, math


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
    acc = primitives.variable(p.get_value() * 0.)
    acc_new = rho * acc + (1 - rho) * g ** 2
    gradient_scaling = math.sqrt(acc_new + epsilon)
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
    v = primitives.variable(p.get_value() * 0.)
    v_next = momentum * v - weight_decay * lr * p - lr * g
    updates.append((v, v_next))

    updates.append((p, p + v_next))

  return updates
