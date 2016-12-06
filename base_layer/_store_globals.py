""" Utility for getting and storing state accross the entire package. """


# The backend module to use for computations.
backend = None
# The name of the backend module.
backend_name = ""
# The theano-specific utilities module.
theano_utils = None

# Random number generator to use for computations.
random = None

# The session we're using. (For Tensorflow.)
session = None
# The savers to use for variable auto-registration.
savers = []


def check_backend():
  """ Checks that backend parameters are actually set. """
  # We're going to force the user to manually import primitives and set the
  # backend before importing this module.
  if not backend:
    raise RuntimeError( \
        "Please call backend.set_backend() before using this module.")
