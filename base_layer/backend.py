""" User-accessible utility for controlling the backend. """


import logging

import _store_backend


logger = logging.getLogger()


def set_backend(backend):
  """ Sets the backend to use. This must be called at some point for this module
  to be useful.
  Args:
    backend: Either "tensorflow" or "theano". """
  logger.info("Using backend '%s'." % (backend))

  if backend == "theano":
    # Import Theano modules.
    import theano
    import theano.tensor
    import theano.ifelse

    _store_backend.backend = theano

    # Local theano modules, which require _backend to be set upon importing.
    from .. theano_layer import utils
    _store_backend.theano_utils = utils

  elif backend == "tensorflow":
    # Import tensorflow modules.
    import tensorflow

    _store_backend.backend = tensorflow

  else:
    raise ValueError("Invalid backend choice: '%s'." % (backend))

  _store_backend.backend_name = backend
