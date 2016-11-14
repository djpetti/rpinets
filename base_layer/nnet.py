""" Functions specific to neural networks. """


import logging

from . import _store_backend as sb


sb.check_backend()

logger = logging.getLogger(__name__)


def dropout(layer_output, keep_prob, is_training, noise_shape=None, seed=None,
            name=None):
  """ Computes dropout for a layer.
  Args:
    layer_output: The activations to perform dropout on.
    keep_prob: The probability of keeping any specific element.
    is_training: A symbolic variable that keeps track if we are training or not.
                 If we are not training, dropout will be disabled.
    noise_shape: 1-D tensor representing the shape of generated keep/drop flags.
    seed: Seed for random number generation. This is only used in Tensorflow.
    name: The name of this operation. This is only used in Tensorflow.
  Returns:
    A tensor of the same shape as the input with dropout computed. """
  if sb.backend_name == "theano":
    if not noise_shape:
      # Just use the size of the input.
      noise_shape = layer_output.shape

    # Seed the random number generator if we need to.
    if seed:
      logger.debug("Seeding random number generator with %d" % (seed))
      sb.random.seed(seed)

    distribution = sb.random.binomial(size=noise_shape, p=keep_prob)
    dropped_out = sb.backend.tensor.switch(distribution, layer_output, 0)
    return sb.backend.ifelse.ifelse(is_training, dropped_out, layer_output * 0.5)

  elif sb.backend_name == "tensorflow":
    dropped_out = sb.backend.nn.dropout(layer_output, keep_prob,
                                        noise_shape=noise_shape,
                                        seed=seed, name=name)

    # Ignore it if we're training.
    return sb.backend.cond(is_training, dropped_out, layer_output * 0.5)

def relu(features, name=None):
  """ Computes the ReLU function.
  Args:
    features: The feature tensor to compute ReLU on.
    name: The name of this operation. This is only used in Tensorflow.
  Returns:
    The computed ReLU tensor. """
  if sb.backend_name == "theano":
    return sb.backend.tensor.nnet.relu(features)
  elif sb.backend_name == "tensorflow":
    return sb.backend.nnet.relu(features, name=name)

def softmax(logits, name=None):
  """ Computes the softmax of a tensor.
  Args:
    logits: The tensor to compute the softmax of.
  Returns:
    The computed softmax. """
  if sb.backend_name == "theano":
    return sb.backend.tensor.nnet.softmax(logits)
  elif sb.backend_name == "tensorflow":
    return sb.backend.nn.softmax(logits, name=name)

def softmax_cross_entropy(logits, labels, name=None):
  """ Computes the cross-entropy of the prediction with the expected outputs.
  It also performs the softmax internally for scaling.
  Args:
    logits: The actual outputs.
    labels: The expected outputs.
  Returns:
    The computed cross-entropy. """
  if sb.backend_name == "theano":
    scaled = softmax(logits)
    return sb.backend.tensor.nnet.categorical_crossentropy(scaled, labels)

  elif sb.backend_name == "tensorflow":
    return sb.backend.nn.softmax_cross_entropy_with_logits(logits, labels,
                                                           name=name)
