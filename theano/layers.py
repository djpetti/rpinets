""" Contains simple classes for representing layers. """

class _WeightLayer(object):
  """ Superclass shared by layers that have weights and biases. """

  def __init__(self, *args, **kwargs):
    # Initial value for our bias.
    self.start_bias = kwargs.get("start_bias", 0)
    # Method for initializing weights. The options are 'xavier' and 'gaussian'.
    self.weight_init = kwargs.get("weight_init", "xavier")
    # If weight_init is set to gaussian, these parameters control the
    # distribution's mean and standard deviation.
    self.weight_mean = kwargs.get("weight_mean", 0)
    self.weight_stddev = kwargs.get("weight_stddev", 1)

class ConvLayer(_WeightLayer):
  """ A simple class to handle the specification of convolutional layers. """

  def __init__(self, *args, **kwargs):
    super(ConvLayer, self).__init__(*args, **kwargs)

    # Convolutional kernel width.
    self.kernel_width = kwargs.get("kernel_width")
    # Convolutional kernel height.
    self.kernel_height = kwargs.get("kernel_height")
    # Number of output feature maps.
    self.feature_maps = kwargs.get("feature_maps")
    # Stride size for convolution. (Defaults to (1, 1))
    self.stride_width = kwargs.get("stride_width", 1)
    self.stride_height = kwargs.get("stride_height", 1)
    # Border mode for convolution. Currently supports either "valid" or
    # "half".
    self.border_mode = kwargs.get("border_mode", "valid")
    # Initial value for our bias.
    self.start_bias = kwargs.get("start_bias", 0)

class PoolLayer(object):
  """ A simple class to handle the specification of maxpooling layers. """

  def __init__(self, *args, **kwargs):
    # Maxpooling kernel size. (Defaults to 2x2.)
    self.kernel_width = kwargs.get("kernel_width", 2)
    self.kernel_height = kwargs.get("kernel_height", 2)
    # Maxpooling stride size. (Defaults to the same as the kernel.)
    self.stride_width = kwargs.get("stride_width", self.kernel_width)
    self.stride_height = kwargs.get("stride_height", self.kernel_height)

class NormalizationLayer(object):
  """ Performs local response normalization, as described in the AlexNet
  paper. """

  def __init__(self, *args, **kwargs):
    self.depth_radius = kwargs.get("depth_radius", 5)
    self.alpha = kwargs.get("alpha", 1.0)
    self.beta = kwargs.get("beta", 0.5)
    self.bias = kwargs.get("bias", 1.0)

class InnerProductLayer(_WeightLayer):
  """ Basic inner product layer. """

  def __init__(self, *args, **kwargs):
    super(InnerProductLayer, self).__init__(*args, **kwargs)

    # Number of neurons in the layer.
    self.size = kwargs.get("size")
    # Whether to use dropout on this layer.
    self.dropout = kwargs.get("dropout", False)
    # Initial value for our bias.
    self.start_bias = kwargs.get("start_bias", 0)
