import logging

from .primitives import _backend, _backend_name, _utils


# We're going to force the user to manually import primitives and set the
# backend before importing this module.
if not _backend:
  raise RuntimeError( \
      "Please call primitives.set_backend() before using this module.")


logger = logging.getLogger(__name__)


class _Optimizer(object):
  """ Represents something that can perform some kind of optimization on the
  graph. Tensorflow has native Optimizer types that this is equivalent to, and
  for Theano, it's a high-level wrapper around more basic functionality. """

  def __init__(self, to_optimize, **kwargs):
    """
    Args:
      to_optimize: The symbolic value to optimize. """
    self._to_optimize = to_optimize

  def native(self):
    """ Must be implemented by a subclass.
    Returns:
      The output of the optimizer, in symbolic form. """
    raise NotImplementedError("'native' must be implemented by subclass.")

  def get_updates(self):
    """ This function is only used by Theano. When a function needs to be
    created, it allows whatever's creating that function to obtain the proper
    updates to use in order for this optimizer to work correctly. """
    if _backend_name == "tensorflow":
      raise NotImplementedError("'get_updates' is only available with Theano.")

    return self._updates

class GradientDescentOptimizer(_Optimizer):
  """ Optimizes a graph using SGD. """

  def __init__(self, learning_rate, *args, **kwargs):
    """
    See documentation for the superclass method.
    Additional Args:
      learning_rate: The learning rate to use.
      momentum: The momentum to use.
      weight_decay: The weight decay to use.
      params: The parameters to update. With Tensorflow, this is technically not
              needed, but it doesn't hurt to provide it anyway. """
    super(GradientDescentOptimizer, self).__init__(*args, **kwargs)

    momentum = kwargs.get("momentum", 0)
    weight_decay = kwargs.get("weight_decay", 0)
    params = kwargs.get("params")

    if _backend_name == "theano":
      if not params:
        raise ValueError("Need 'params' argument with Theano.")

      # Compute the updates to use.
      self._updates = _utils.momentum_sgd(self._to_optimize, params,
                                          learning_rate, momentum,
                                          weight_decay)

    elif _backend_name == "tensorflow":
      # Use the built-in optimizer.
      # TODO (danielp): Implement weight decay in Tensorflow.
      self.__optimizer = _backend.MomentumOptimizer(learning_rate, momentum)

  def native(self):
    """ See documentation for superclass method. """
    if _backend_name == "theano":
      # Just return the training cost here, which might be useful.
      return self._to_optimize
    elif _backend_name == "tensorflow":
      return self.__optimizer
