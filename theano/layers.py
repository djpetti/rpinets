""" Contains simple classes for representing layers. """


class ConvLayer(object):
  """ A simple class to handle the specification of convolutional layers. """

  def __init__(self, *args, **kwargs):
    # Convolutional kernel width.
    self.kernel_width = kwargs.get("kernel_width")
    # Convolutional kernel height.
    self.kernel_height = kwargs.get("kernel_height")
    # Number of input feature maps.
    self.feature_maps = kwargs.get("feature_maps")
    # Stride size for convolution. (Defaults to (1, 1))
    self.stride_width = kwargs.get("stride_width", 1)
    self.stride_height = kwargs.get("stride_height", 1)
    # Border mode for convolution. Currently supports either "valid" or
    # "half".
    self.border_mode = kwargs.get("border_mode", "valid")

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

class InnerProductLayer(object):
  """ Basic inner product layer. """

  def __init__(self, *args, **kwargs):
    # Number of neurons in the layer.
    self.size = kwargs.get("size")
    # Whether to use dropout on this layer.
    self.dropout = kwargs.get("dropout", False)
