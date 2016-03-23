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
