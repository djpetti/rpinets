import numpy as np

import logging

from . import _store_globals as sg
from . import primitives


logger = logging.getLogger(__name__)

sg.check_backend()


def square(value, name=None):
  """ Squares a value.
  Args:
    value: The value to square.
    name: Optionally, the name of the squared value. This is ignored for Theano.
  Returns:
    The squared value. """
  if sg.backend_name == "theano":
    return sg.backend.tensor.sqr(value)
  elif sg.backend_name == "tensorflow":
    return sg.backend.square(value, name=name)

def sqrt(value, name=None):
  """ Takes the square root of a value.
  Args:
    value: The value to take the square root of.
    name: Optionally, the name of the new value. This is ignored for Theano.
  Returns:
    The square root of value. """
  if sg.backend_name == "theano":
    return sg.backend.tensor.sqrt(value)
  elif sg.backend_name == "tensorflow":
    return sg.backend.sqrt(value, name=name)

def dot(matrix1, matrix2, name=None):
  """ Computes the dot product of two matrices.
  Args:
    matrix1: The first matrix to use.
    matrix2: The second matrix to use.
    name: Optionally, the name of the new value. This is ignored for Theano.
  Returns:
    The dot product. """
  if sg.backend_name == "theano":
    return sg.backend.tensor.dot(matrix1, matrix2)
  elif sg.backend_name == "tensorflow":
    return sg.backend.matmul(matrix1, matrix2, name=None)

def mean(tensor, axis=None, keepdims=False, name=None):
  """ Computes the mean of a tensor along some axis.
  Args:
    tensor: The tensor to compute the mean of.
    axis: The axis to compute the mean along. If it is None, every axis will be
          used, like Numpy. If it is a list, those axes will be used. If it is
          an int, only that axis will be used.
    keepdims: Whether to keep reduced dimensions with length 1 or not.
    name: Name of the new value. This is ignored for Theano.
  Returns:
    The computed mean. """
  if type(axis) == int:
    # Tensorflow doesn't support this being an int, so we just always make it a
    # list.
    axis = [axis]

  if sg.backend_name == "theano":
    return sg.backend.tensor.mean(tensor, axis=axis, keepdims=keepdims)
  elif sg.backend_name == "tensorflow":
    return sg.backend.reduce_mean(tensor, reduction_indices=axis,
                                  keep_dims=keepdims, name=name)

def argmax(tensor, axis, name=None):
  """ Computes the index of the max value along an axis.
  Args:
    tensor: The tensor to compute the argmax of.
    axis: The axis to compute the argmax on.
    name: Name of the new value. This is ignored for Theano.
  Returns:
    The index of the max value along axis. """
  if sg.backend_name == "theano":
    return sg.backend.tensor.argmax(tensor, axis=axis)
  elif sg.backend_name == "tensorflow":
    return sg.backend.argmax(tensor, axis, name=name)

def equal(tensor1, tensor2, name=None):
  """
  Returns:
    The truth value of tensor1 == tensor2 elementwise. """
  if sg.backend_name == "theano":
    return sg.backend.tensor.eq(tensor1, tensor2)
  elif sg.backend_name == "tensorflow":
    return sg.backend.equal(tensor1, tensor2, name=name)

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
  return primitives.cast(rate, "float32")

def flatten(tensor, outdim=1):
  """ Flattens a tensor into a vector.
  Args:
    tensor: The tensor to flatten.
    outdim: The number of dimensions in the result. It defaults to 1.
  Returns:
    The flattened tensor. """
  if sg.backend_name == "theano":
    return sg.backend.tensor.flatten(tensor, outdim=outdim)

  elif sg.backend_name == "tensorflow":
    # Tensorflow doesn't actually have a native flattening op, but we can fake
    # it easily.
    shape = tensor.get_shape().as_list()
    new_dim_size = np.prod(shape[outdim - 1:])

    new_shape = [-1] * (outdim - 1) + [new_dim_size]
    logger.debug("Reshaping to: %s" % (new_shape))
    return sg.backend.reshape(tensor, new_shape)
