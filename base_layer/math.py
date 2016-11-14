from . import _store_backend as sb


sb.check_backend()


def square(value, name=None):
  """ Squares a value.
  Args:
    value: The value to square.
    name: Optionally, the name of the squared value. This is ignored for Theano.
  Returns:
    The squared value. """
  if sb.backend_name == "theano":
    return sb.backend.tensor.sqr(value)
  elif sb.backend_name == "tensorflow":
    return sb.backend.square(value, name=name)

def sqrt(value, name=None):
  """ Takes the square root of a value.
  Args:
    value: The value to take the square root of.
    name: Optionally, the name of the new value. This is ignored for Theano.
  Returns:
    The square root of value. """
  if sb.backend_name == "theano":
    return sb.backend.tensor.sqrt(value)
  elif sb.backend_name == "tensorflow":
    return sb.backend.sqrt(value, name=name)

def dot(matrix1, matrix2, name=None):
  """ Computes the dot product of two matrices.
  Args:
    matrix1: The first matrix to use.
    matrix2: The second matrix to use.
    name: Optionally, the name of the new value. This is ignored for Theano.
  Returns:
    The dot product. """
  if sb.backend_name == "theano":
    return sb.backend.tensor.dot(matrix1, matrix2)
  elif sb.backend_name == "tensorflow":
    return sb.backend.matmul(matrix1, matrix2, name=None)

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

  if sb.backend_name == "theano":
    return sb.backend.tensor.mean(tensor, axis=axis, keepdims=keepdims)
  elif sb.backend_name == "tensorflow":
    return sb.backend.reduce_mean(tensor, reduction_indices=axis,
                                  keep_dims=keepdims, name=name)

def argmax(tensor, axis, name=None):
  """ Computes the index of the max value along an axis.
  Args:
    tensor: The tensor to compute the argmax of.
    axis: The axis to compute the argmax on.
    name: Name of the new value. This is ignored for Theano.
  Returns:
    The index of the max value along axis. """
  if sb.backend_name == "theano":
    return sb.backend.tensor.argmax(tensor, axis=axis)
  elif sb.backend_name == "tensorflow":
    return sb.backend.argmax(tensor, axis, name=name)

def equal(tensor1, tensor2, name=None):
  """
  Returns:
    The truth value of tensor1 == tensor2 elementwise. """
  if sb.backend_name == "theano":
    return sb.backend.tensor.eq(tensor1, tensor2)
  elif sb.backend_name == "tensorflow":
    return self.backend.equal(tensor1, tensor2, name=name)
