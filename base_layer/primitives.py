""" Basic functions that are implemented by Tensorflow or Theano. This presents
a unified API for them. """


import logging

from . import _store_globals as sg
from . import saver


sg.check_backend()

logger = logging.getLogger(__name__)


# The following functions are wrappers for creating native types in any library.
# They're meant to be used almost exactly as one would initialize the native
# class.


def placeholder(dtype, shape, name=None):
  """ This is equivalent to a TensorType in theano. In Tensorflow, it's
  equivalent to a Placeholder. It is mainly meant to be used as inputs for the
  graph, where the values are not known at the time the graph is defined.
  Args:
    dtype: The type to use for this placeholder.
    shape: Specifies the shape of the placeholder. Members of this can be None
           to indicate a dimension of unspecified size.
    name: Optionally gives a name to this placeholder.
  Returns:
    The native placeholder type that was created. """
  if sg.backend_name == "theano":
    # Get the broadcasting right.
    broadcastable = [False] * len(shape)

    # Implement as a TensorType.
    use_type = sg.backend.tensor.TensorType(dtype, broadcastable)
    return use_type(name)

  elif sg.backend_name == "tensorflow":
    # Implement with a placeholder.
    return sg.backend.placeholder(dtype, shape=shape, name=name)

def variable(initial_value, name=None):
  """ A class representing a shared variable, which can be stored in GPU memory
  and used in the graph. This is equivalent to theano.shared and
  tensorflow.Variable.
  Args:
    initial_value: The initial value of the variable.
    name: An optional name for the variable.
  Returns:
    The native variable type that was created. """
  variable = None
  if sg.backend_name == "theano":
    variable = sg.backend.shared(initial_value, name=name)
  elif sg.backend_name == "tensorflow":
    variable = sg.backend.Variable(initial_value, name=name)

  # Register the variable with savers.
  saver.VariableSaver.register_with_all(variable)
  return variable

def cast(value, dtype, name=None):
  """ Converts a value to the specified type, symbolically.
  Args:
    value: The value to convert.
    dtype: The type to convert the value into.
    name: Optionally, the name of the casted value. This is ignored for
          Theano.
  Returns:
    The converted value. """
  if sg.backend_name == "theano":
    return sg.backend.tensor.cast(value, dtype)
  elif sg.backend_name == "tensorflow":
    return sg.backend.cast(value, dtype, name=name)
