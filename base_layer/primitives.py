""" Basic functions that are implemented by Tensorflow or Theano. This presents
a unified API for them. """


import logging


logger = logging.getLogger(__name__)


# This keeps track of the backend we are using.
_backend = None
_backend_name = ""
_utils = None

def set_backend(backend):
  """ Sets the backend to use. This must be called at some point for this module
  to be useful.
  Args:
    backend: Either "tensorflow" or "theano". """
  global _backend
  global _backend_name
  global _utils

  logger.info("Using backend '%s'." % (backend))

  if backend == "theano":
    # Import Theano modules.
    import theano
    import theano.tensor
    import theano.ifelse

    # Local theano modules.
    from .. theano_layer import utils
    _utils = utils

    _backend = theano

  elif backend == "tensorflow":
    # Import tensorflow modules.
    import tensorflow

    _backend = tensorflow

  else:
    raise ValueError("Invalid backend choice: '%s'." % (backend))

  _backend_name = backend


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
  if _backend_name == "theano":
    # Get the broadcasting right.
    broadcastable = [False] * len(shape)

    # Implement as a TensorType.
    use_type = _backend.tensor.TensorType(dtype, broadcastable)
    return use_type(name)

  elif _backend_name == "tensorflow":
    # Implement with a placeholder.
    return _backend.placeholder(dtype, shape=shape, name=name)

def variable(initial_value, name=None):
  """ A class representing a shared variable, which can be stored in GPU memory
  and used in the graph. This is equivalent to theano.shared and
  tensorflow.Variable.
  Args:
    initial_value: The initial value of the variable.
    name: An optional name for the variable.
  Returns:
    The native variable type that was created. """
  if _backend_name == "theano":
    return _backend.shared(initial_value, name=name)
  elif _backend_name == "tensorflow":
    return _backend.Variable(initial_value, name=name)
