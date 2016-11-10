import logging

from .primitives import _backend, _backend_name
from . import optimizer


# We're going to force the user to manually import primitives and set the
# backend before importing this module.
if not _backend:
  raise RuntimeError( \
      "Please call primitives.set_backend() before using this module.")


logger = logging.getLogger(__name__)


class Runnable(object):
  """ Wraps the Tensorflow Session and Theano Function interfaces into a single
  interface that can be used for both. """

  # The session that will be used with Tensorflow for actually running things.
  # There is no need to have any more than one of these.
  _session = None

  def __init__(self, inputs, outputs, givens):
    """
    Args:
      inputs: The input data.
      outputs: A list of tensors in the graph to compute the value of.
      givens: A dictionary of tensors and their values. These should be tensors
      whose symbolic values will not change for the lifetime of the runnable.
    """
    # graph_outputs is a list of the outputs where everything been converted
    # to the native backend version.
    graph_outputs = []
    for output in outputs:
      if isinstance(output, optimizer._Optimizer):
        logger.debug("Found new optimizer: %s" % output)
        graph_outputs.append(output.native())
      else:
        graph_outputs.append(output)

    if _backend_name == "theano":
      logger.info("Creating new function...")

      # Find any Optimizers and compute updates accordingly.
      updates = []
      for output in outputs:
        if isinstance(output, optimizer._Optimizer):
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
      in the contructor. """
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
