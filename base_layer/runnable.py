import logging

from . import _store_globals as sg
from . import optimizer


sg.check_backend()

logger = logging.getLogger(__name__)


class Runnable(object):
  """ Wraps the Tensorflow Session and Theano Function interfaces into a single
  interface that can be used for both. """

  def __init__(self, inputs, outputs, givens):
    """
    Args:
      inputs: The input data.
      outputs: A list of tensors in the graph to compute the value of.
      givens: A dictionary of tensors and their values. These should be tensors
      whose symbolic values will not change for the lifetime of the runnable.
    """
    # graph_outputs is a list of the outputs where everything been converted
    # to the native sg.backend version.
    graph_outputs = []
    for output in outputs:
      if isinstance(output, optimizer._Optimizer):
        logger.debug("Found new optimizer: %s" % output)
        graph_outputs.append(output.native())
      else:
        graph_outputs.append(output)

    if sg.backend_name == "theano":
      logger.info("Creating new function...")

      # Find any Optimizers and compute updates accordingly.
      updates = []
      for output in outputs:
        if isinstance(output, optimizer._Optimizer):
          updates.extend(output.get_updates())

      self.__function = sg.backend.function(inputs=inputs, outputs=graph_outputs,
                                            givens=givens, updates=updates)

    elif sg.backend_name == "tensorflow":
      if not sg.session:
        # We still need to make a global session.
        logger.info("Creating session...")
        sg.session = sg.backend.Session()

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
    if sg.backend_name == "theano":
      # Just run the function.
      return self.__function(*input_values)

    elif sg.backend_name == "tensorflow":
      # In Tensorflow, the givens are technically just members of the feed dict
      # that we've specified ahead-of-time.
      feed_dict = self.__givens
      # Add the newly-specified inputs.
      for tensor, value in zip(self.__inputs, input_values):
        feed_dict[tensor] = value
      logger.debug("Feeding with %s" % (feed_dict))

      # We have to initialize variables.
      # TODO (danielp): Be smarter about doing this only when we have to.
      sg.session.run(sg.backend.global_variables_initializer())

      # Now, we can just run our session.
      return sg.session.run(self.__outputs, feed_dict=feed_dict)