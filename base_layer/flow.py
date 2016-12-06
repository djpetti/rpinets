""" Functions for controlling flow in the graph. """


import logging

from . import _store_globals as sg
from . import primitives


sg.check_backend()

logger = logging.getLogger(__name__)


def ifelse(condition, if_true, if_false):
  """ Adds a conditional branch to the graph.
  Args:
    condition: The condition to check.
    if_true: What to return if condition is true.
    if_false: What to return if condition is false.
  Returns:
    Conditional graph operation. """
  if sg.backend_name == "theano":
    return sg.backend.ifelse.ifelse(condition, if_true, if_false)
  elif sg.backend_name == "tensorflow":
    # Theano doesn't have a bool type, so everything must be passed as ints for
    # compatibility. That means we have to manually cast for Tensorflow.
    condition = primitives.cast(condition, "bool")

    return sg.backend.cond(condition, lambda: if_true, lambda: if_false)
