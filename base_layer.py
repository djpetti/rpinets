""" Basic functions that are implemented by Tensorflow or Theano. This presents
a unified API for them. """


import logging


logger = logging.getLogger(__name__)


# This keeps track of the backend we are using.
_backend = None
_backend_name = ""


def set_backend(backend):
  """ Sets the backend to use. This must be called at some point for this module
  to be useful.
  Args:
    backend: Either "tensorflow" or "theano". """
  global _backend
  global _backend_name

  logger.info("Using backend '%s'." % (backend))

  if backend == "theano":
    # Import Theano modules.
    import theano
    import theano.tensor
    import theano.ifelse

    # Local theano modules.
    from .theano_layer import utils

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


class Runnable(object):
  """ Wraps the Tensorflow Session and Theano Function interfaces into a single
  interface that can be used for both. """

  # The session that will be used with Tensorflow for actually running things.
  # There is no need to have any more than one of these.
  _session = None
  # Keeps track of a global step count.
  _global_step = None

  def __init__(self, inputs, outputs, givens):
    """
    Args:
      inputs: The input data.
      outputs: A list of tensors in the graph to compute the value of.
      givens: A dictionary of tensors and their values. These should be tensors
      whose symbolic values will not change for the lifetime of the runnable.
    """
    # Create a global step variable if we need to.
    if not Runnable._global_step:
      logger.debug("Starting global step at 0.")
      Runnable._global_step = variable(0, name="global_step")
    # Symbolic next global step value.
    next_step = Runnable._global_step + 1
    # Add updated global step to the outputs.
    outputs.append(next_step)

    # graph_outputs is a list of the outputs where everything been converted
    # to the native backend version.
    graph_outputs = []
    for output in outputs:
      if isinstance(output, _Optimizer):
        logger.debug("Found new optimizer: %s" % output)
        graph_outputs.append(output.native())
      else:
        graph_outputs.append(output)

    if _backend_name == "theano":
      logger.info("Creating new function...")

      # Find any Optimizers and compute updates accordingly.
      updates = []
      for output in outputs:
        if isinstance(output, _Optimizer):
          updates.extend(output.get_updates())

      self.__function = _backend.function(inputs=inputs, outputs=graph_outputs,
                                          givens=givens, updates=updates)

    elif _backend_name == "tensorflow":
      if not Runnable._session:
        # We still need to make a global session.
        logger.info("Creating session...")
        Runnable._session = _backend.Session()

      # We can't do anything right now, so just save everything.
      self.__inputs = inputs
      self.__outputs = graph_outputs
      self.__givens = givens

  def run(self, *input_values):
    """ Runs a single step with the inputs to compute the outputs.
    Args:
      input_values: These arguments will be the values given to the input
                    parameters, in the order that said input parameters were
                    specified in the constructor.
    Returns:
      The computed output values, in the order that said outputs were specified
      in the contructor. The last output will always be the global step
      counter, which increments every time run is called. """
    if _backend_name == "theano":
      # Just run the function.
      return self.__function(*input_values)

    elif _backend_name == "tensorflow":
      # In Tensorflow, the givens are technically just members of the feed dict
      # that we've specified ahead-of-time.
      feed_dict = self.__givens
      # Add the newly-specified inputs.
      for tensor, value in zip(self.__inputs, input_values):
        feed_dict[tensor] = value
      logger.debug("Feeding with %s" % (feed_dict))

      # Now, we can just run our session.
      return Runnable._session.run(self.__outputs, feed_dict=feed_dict)


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
    super(_GradientDescentOptimizer, self).__init__(*args, **kwargs)

    momentum = kwargs.get("momentum", 0)
    weight_decay = kwargs.get("weight_decay", 0)
    params = kwargs.get("params")

    if _backend_name == "theano":
      if not params:
        raise ValueError("Need 'params' argument with Theano.")

      # Compute the updates to use.
      self._updates = utils.momentum_sgd(self._to_optimize, params,
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
