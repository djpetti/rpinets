import json


def make_img_id(label, name):
  """ Creates the image ID for an image.
  Args:
    label: The image label.
    name: The name of the image within the label.
  Returns:
    The image ID. """
  return json.dumps([label, name])

def split_img_id(img_id):
  """ Splits an image ID into the image name and label.
  Args:
    img_id: The image ID to split.
  Returns:
    The image label and name. """
  return json.loads(img_id)
