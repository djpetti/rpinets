""" Classes that simplify learning rate modification. """


import theano
import theano.tensor as TT


class _LearningRate(object):
  """ Suplerclass for learning rates. """

  def __init__(self, initial_rate):
    """
    Args:
      initial_rate: Initial value of the learning rate. """
    self._rate = initial_rate

  def get(self, cycle):
    """
    Args:
      cycle: The symbolic global step.
    Returns:
      The learning rate to use for this cycle. """
    return TT.as_tensor_variable(self._rate, name="lr")

class Fixed(_LearningRate):
  """ The simplest type of learning rate. It is just a fixed value. """
  pass

class ExponentialDecay(_LearningRate):
  """ A learning rate that decays exponentially with time. """

  def __init__(self, decay_rate, decay_steps, *args, **kwargs):
    """
    Args:
      decay_rate: Number of steps needed to decay by decay_rate.
      decay_steps: The decay rate. """
    super(ExponentialDecay, self).__init__(*args, **kwargs)

    self.__decay_steps = decay_steps
    self.__decay_rate = decay_rate

  def get(self, cycle):
    rate = self._rate * self.__decay_rate ** (cycle / float(self.__decay_steps))
    return TT.cast(rate, theano.config.floatX)
