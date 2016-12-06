""" Abstracts away differences in how we save data to files. """


import cPickle as pickle
import logging
import os

from . import _store_globals as sg
from . import runnable


logger = logging.getLogger(__name__)

sg.check_backend()


class VariableSaver(object):
  """ Common interface for saving variable data. """

  def __init__(self, auto_register=True):
    """
    Args:
      auto_register: If true, any new variables created will be automatically
                     registered with this saver. Otherwise, the only way to
                     register variables is by manually calling the register()
                     method. """
    logger.info("Creating new saver.")
    if auto_register:
      sg.savers.append(self)

    # List of variables that are being managed.
    self.__variables = []

  def register(self, variable):
    """ Registers a new variable with this saver. """
    logger.debug("Registering '%s' with saver." % (variable))

    self.__variables.append(variable)

  @classmethod
  def register_with_all(cls, variable):
    """ Registers a variable with all savers. """
    if not sg.savers:
      print "No registered savers."
    for saver in sg.savers:
      saver.register(variable)

  def save(self, path):
    """ Saves all registered variables to a file.
    Args:
       path: The prefix of the path to save data to. """
    logger.debug("Saving %d variables." % (len(self.__variables)))

    if sg.backend_name == "theano":
      # We're going to mark this as a Theano file to ensure that Tensorflow
      # doesn't try to open it.
      path += "_theano.ckpt"

      # In Theano, everything is Pickleable, so we can just use that to
      # implement saving.
      save_to = open(path, "wb")
      pickle.dump(self.__variables, save_to, protocol=pickle.HIGHEST_PROTOCOL)
      save_to.close()

    elif sg.backend_name == "tensorflow":
      # Use the built-in saver class to handle saving.
      saver = sg.backend.train.Saver(var_list=self.__variables)
      saver.save(sg.session, path)

  def load(self, path):
    """ Loads all registered variables from a file. For Theano at least, it
    raises an exception if the variables loaded do not match those currently
    registered with the saver.
    Args:
      path: The prefix of the path to load from. """
    if sg.backend_name == "theano":
      # We marked this as a Theano file when we saved it.
      path += "_theano.ckpt"

      load_from = file(path, "rb")
      loaded_variables = pickle.load(load_from)
      if len(loaded_variables) != len(self.__variables):
        raise ValueError("Loaded variables do not match registered variables!")

      # Set existing variables accordingly.
      for variable, loaded_variable in zip(self.__variables, loaded_variables):
        variable.set_value(loaded_variable.get_value())

    if sg.backend_name == "tensorflow":
      # Before we can do anything, we've got to get the filename from
      # Tensorflow.
      path = os.path.dirname(path)
      if not path:
        # In this case, it's just the current directory.
        path = "."

      latest_checkpoint = sg.backend.train.latest_checkpoint(path)
      if not latest_checkpoint:
        raise ValueError("Could not find saved data for path '%s'." % (path))
      logger.debug("Loading saved checkpoint: %s" % (latest_checkpoint))

      saver = sg.backend.train.Saver(var_list=self.__variables)
      saver.restore(sg.session, latest_checkpoint)
