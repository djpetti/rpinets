import random


class RandomSet(object):
  """ A special set that sacrifices memory but allows for O(1) acquisition of
  random elements. """

  def __init__(self):
    # Holds an ordered version of the data for easy random picking.
    self.__data = []
    # Maps data in the set to indices in __data.
    self.__data_to_indices = {}

  def __init__(self, initial_data):
    """ Initializes the set from some initial data.
    Args:
      initial_data: A list of the initial data. """
    self.__data = initial_data

    self.__data_to_indices = {}
    for i, item in enumerate(initial_data):
      self.__data_to_indices[item] = i

  def add(self, item):
    """ Adds a new item to the set.
    Args:
      item: The item to add. """
    if item not in self:
      # If it's not a duplicate, add it.
      self.__data.append(item)

      index = len(self.__data) - 1
      self.__data_to_indices[item] = index

  def remove(self, item):
    """ Remove an item from the set.
    Args:
      item: The item to remove. """
    index = self.__data_to_indices[item]
    self.__data_to_indices.pop(item)

    # Replace this element with whatever's in the last slot.
    if index == len(self.__data) - 1:
      # Special case: We're removing the last element.
      self.__data.pop()
    else:
      new_item = self.__data.pop()
      self.__data[index] = new_item
      self.__data_to_indices[new_item] = index

  def get_random(self):
    """ Returns: A random element from the set, in O(1) time. """
    index = random.randint(0, len(self.__data) - 1)
    return self.__data[index]

  def union(self, other):
    """ Take the union of two RandomSets.
    Args:
      other: The other RandomSet.
    Returns:
      The union of the two sets. """
    union = RandomSet(self.__data[:])
    for item in other.__data:
      union.add(item)

    return union

  def __len__(self):
    return len(self.__data)

  def __contains__(self, item):
    return item in self.__data_to_indices

  def __iter__(self):
    return self.__data.__iter__()
