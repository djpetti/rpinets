""" Utility for getting and storing the backend. """


# The backend module to use for computations.
backend = None
# The name of the backend module.
backend_name = ""
# The theano-specific utilities module.
theano_utils = None


def check_backend():
  """ Checks that backend parameters are actually set. """
  # We're going to force the user to manually import primitives and set the
  # backend before importing this module.
  if not backend:
    raise RuntimeError( \
        "Please call backend.set_backend() before using this module.")
