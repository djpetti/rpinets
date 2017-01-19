import theano
import theano.tensor as TT

import numpy as np

from simple_lenet import LeNetClassifier
import layers


class AlexNet(LeNetClassifier):
  """ Special class specifically for Alexnets. It handles the initialization and
  special testing methods. """

  def __init__(self, train, test, batch_size, layer_override=None, **kwargs):
    """
    Args:
      train: The training dataset, same as for LeNetClassifier.
      test: The testing dataset, same as for LeNetClassifier.
      batch_size: The size of each batch.
      layer_override: Allows users to specify a custom set of layers instead of
      the default AlexNet ones. """
    self.__softmaxes = []

    self.__backwards_propagator = None

    # Initialize layers.
    if not layer_override:
      conv1 = layers.ConvLayer(kernel_width=11, kernel_height=11, stride_width=4,
                              stride_height=4, feature_maps=96,
                              border_mode="half")
      conv2 = layers.ConvLayer(kernel_width=5, kernel_height=5, feature_maps=256,
                              border_mode="half", start_bias=1)
      conv3 = layers.ConvLayer(kernel_width=3, kernel_height=3, feature_maps=384,
                              border_mode="half")
      conv4 = layers.ConvLayer(kernel_width=3, kernel_height=3, feature_maps=384,
                              border_mode="half", start_bias=1)
      conv5 = layers.ConvLayer(kernel_width=3, kernel_height=3, feature_maps=256,
                              border_mode="half", start_bias=1)
      pool = layers.PoolLayer(kernel_width=3, kernel_height=3, stride_width=2,
                              stride_height=2)
      flatten = layers.InnerProductLayer(size=6 * 6 * 256, dropout=True,
                                        start_bias=1, weight_init="gaussian",
                                        weight_stddev=0.005)
      inner_product1 = layers.InnerProductLayer(size=4096, dropout=True,
                                                start_bias=1,
                                                weight_init="gaussian",
                                                weight_stddev=0.005)
      inner_product2 = layers.InnerProductLayer(size=4096, weight_init="gaussian",
                                                weight_stddev=0.005)
      norm = layers.NormalizationLayer(depth_radius=5, alpha=1e-05 ,beta=0.75,
                                       bias=1.0)

      use_layers = [conv1, pool, norm, conv2, pool, norm, conv3, conv4, conv5, pool,
                    flatten, inner_product1, inner_product2]
    else:
      use_layers = layer_override

    super(AlexNet, self).__init__((224, 224, 3), use_layers, 1000, train, test,
                                  batch_size, **kwargs)

  def _build_tester(self, test_x, test_y, batch_size):
    """ Same as the superclass tester, but returns the raw softmax
    instead of the accuracy. """
    index = TT.lscalar()
    softmax = TT.nnet.softmax(self._layer_stack)

    batch_start = index * batch_size
    batch_end = (index + 1) * batch_size
    tester = theano.function(inputs=[index], outputs=softmax,
                             givens={self._inputs: \
                                     test_x[batch_start:batch_end],
                                     self._training: 0})
    return tester

  def __accuracy_from_softmax(self, softmax, expected_outputs):
    """ Computes the top-one and top-five accuracies given a softmax
    distribution.
    Args:
      softmax: The softmax output from the network.
      expected_outputs: A non-symbolic copy of our expected outputs.
    Returns:
      The top-one and top-five accuracy. """
    # Now find the accuracy.
    sort = np.argsort(softmax, axis=1)
    top_one = sort[:, -1:]
    top_five = sort[:, -5:]
    top_one_accuracy = np.mean(np.equal(expected_outputs,
                                        np.transpose(top_one)))

    # Top five accuracy.
    correct = 0
    for i in range(0, self._batch_size):
      if np.in1d(expected_outputs[i], top_five[i])[0]:
        correct += 1
    top_five_accuracy = float(correct) / self._batch_size

    return top_one_accuracy, top_five_accuracy

  def __get_mean_softmax(self, batch_index, patches=10):
    """ Computes and averages the softmax over a set of patches.
    Args:
      batch_index: The index of the first batch to use.
      patches: The number of patches to average accross.
    Returns:
      The averaged softmax distribution. """
    # Since the tester does everything in terms of batch size, we need to
    # convert patch_separation to batch sized units.
    separation = self._patch_separation / self._batch_size
    softmaxes = []
    # Run for every patch.
    for i in range(0, patches * separation, separation):
      softmaxes.append(self._tester(batch_index + i))

    # Find the mean distribution.
    softmaxes = np.asarray(softmaxes)
    mean = np.mean(softmaxes, axis=0)

    return mean

  def test(self, batch_index, expected_outputs, patches=10):
    """ A special tester that averages the softmax accross multiple
    translations, as described in the AlexNet paper. It is assumed that
    different translations of the same batch are stored as sequential batches in
    the dataset.
    Args:
      batch_index: The index of the first batch to use.
      expected_outputs: A non-symbolic copy of our expected outputs.
      patches: The number of patches to average accross.
    Returns:
      The top-one and top-five accuracy of the network. """
    mean = self.__get_mean_softmax(batch_index, patches=patches)
    return self.__accuracy_from_softmax(mean, expected_outputs)

  def predict_patched(self, batch_index, patches=10):
    """ Works like a standard predictor, except that it computes the predictions
    from a softmax distribution averaged accross a number of patches.
    Args:
      batch_index: The index of the first batch to use.
      patches: The number of patches to average accross.
    Returns:
      The predicted labels from the network. """
    mean = self.__get_mean_softmax(batch_index, patches=patches)

    sort = np.argsort(mean, axis=1)
    top_one = sort[:, -1:]
    return np.transpose(top_one)[0]

  def test_patchless(self, batch_index, expected_outputs):
    """ A simple tester that works the same as the superclass version, with no
    patches.
    Args:
      batch_index: The index of the batch to use.
      expected_outputs: A non-symbolic copy of our expected outputs.
    Returns:
      The top-one and top-five accuracy of the network. """
    softmax = super(AlexNet, self).test(batch_index)
    return self.__accuracy_from_softmax(softmax, expected_outputs)

  def l2_norm_backwards(self, index):
    """ A method useful for dreaming. Computes all the top network gradients for
    maximizing the L2 norm of the output layer activations.
    Args:
      index: The index into the testing batch that we will compute gradients
      for. """
    if not self.__backwards_propagator:
      l2 = self._intermediate_activations[-4]

      index_var = TT.lscalar()
      batch_start = index_var * self._batch_size
      batch_end = (index_var + 1) * self._batch_size

      # We want the gradients for the input image.
      params = [self._inputs]

      known_grads = {l2: l2}
      grads = TT.grad(None, wrt=params, known_grads=known_grads)

      self.__backwards_propagator = theano.function(inputs=[index_var],
          outputs=grads, givens={self._inputs: \
                                 self._test_x[batch_start:batch_end]})

    return self.__backwards_propagator(index)

  @classmethod
  def load(cls, *args, **kwargs):
    network = super(AlexNet, cls).load(*args, **kwargs)

    network.__softmaxes = []
    network.__backwards_propagator = None

    return network
