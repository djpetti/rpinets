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
