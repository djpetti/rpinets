import theano
import theano.tensor as TT

import numpy as np

from simple_lenet import LeNetClassifier


class AlexNet(LeNetClassifier):
  """ Special class specifically for Alexnets. """

  def __init__(self, *args, **kwargs):
    super(AlexNet, self).__init__(*args, **kwargs)

  def _build_tester(self, test_x, test_y, batch_size):
    """ Same as the superclass tester, but returns the raw softmax instead of
    the accuracy. """
    index = TT.lscalar()
    softmax = TT.nnet.softmax(self._layer_stack)

    batch_start = index * batch_size
    batch_end = (index + 1) * batch_size
    tester = theano.function(inputs=[index], outputs=softmax,
                             givens={self._inputs: \
                                     test_x[batch_start:batch_end]})
    return tester

  def test(self, batch_index, expected_outputs):
    """ A special tester that averages the softmax accross multiple
    translations, as described in the AlexNet paper. It is assumed that
    different translations of the same batch are stored as sequential batches in
    the dataset.
    Args:
      index: The index of the first batch to use.
      expected_outputs: A non-symbolic copy of our expected outputs.
    Returns:
      The accuracy of the network. """
    # Run for every translation.
    softmaxes = []
    for i in range(0, 5):
      softmaxes.append(self._tester(batch_index + i))

    # Find the mean distribution.
    softmaxes = np.asarray(softmaxes)
    mean = np.mean(softmaxes, axis=0)

    # Now find the accuracy.
    argmax = np.argmax(mean, axis=1)
    # expected_outputs includes duplicate values for each patch.
    accuracy = np.mean(np.equal(expected_outputs[0:128], argmax))

    return accuracy
