""" Class that downloads and processes requested images in separate processes.
"""


import collections
import logging
import multiprocessing
import random
import time

import data_augmentation
import images


logger = logging.getLogger(__name__)


# Number of worker processes we will use.
WORKER_LIMIT = 100


class DownloaderProcess(multiprocessing.Process):
  """ A process that downloads a single image and exits. """
  def __init__(self, command_queue, image_queue, *args, **kwargs):
    """ Args:
      command_queue: Queue that we read download commands from.
      image_queue: Queue that we write image information onto. """
    super(DownloaderProcess, self).__init__(*args, **kwargs)

    self.__command_queue = command_queue
    self.__image_queue = image_queue

  def run(self):
    """ Runs the process. """
    logger.debug("Starting downloader process.")

    while True:
      synset, number, url = self.__command_queue.get()

      self.__download_image(synset, number, url)

  def __download_image(self, synset, number, url):
    """ Downloads a single image.
    Args:
      synset: The image synset.
      number: The image number.
      url: The image url. """
    logger.debug("Downloading new image: %s_%s" % (synset, number))

    # Download the image.
    image = images.download_image(url, keep_color=True)
    if image is None:
      logger.warning("Failed to download %s." % (url))
      self.__image_queue.put((synset, number, url, None, None))
      return

    # We should choose a patch here before it gets put in the buffer.
    patches = data_augmentation.extract_patches(image)

    # Save the image to the queue.
    logger.debug("Saving image: %s_%s" % (synset, number))
    self.__image_queue.put((synset, number, url, image,
                            patches))


class DownloadManager(object):
  """ Deals with managing and dispatching downloads. """

  def __init__(self, disk_cache, mem_buffer,
               all_patches=False):
    """
    Args:
      disk_cache: DiskCache to save downloaded images to.
      mem_buffer: MemoryBuffer to save downloaded images to.
      all_patches: Whether to save all the patches, or just pick one at random.
      """
    self.__disk_cache = disk_cache
    self.__mem_buffer = mem_buffer
    self.__all_patches = all_patches

    # Set of failed downloads.
    self.__failures = set([])

  def download_new(self, synset, number, url):
    """ Registers a new image to be downloaded.
    Args:
      synset: The synset of the image.
      number: The image number in the synset.
      url: The url of the image. """
    # Add a new download.
    _command_queue.put((synset, number, url))

  def update(self):
    """ Adds any new processes, and cleans up any old ones. Should be called
    periodically.
    Returns:
      The number of new images that were successfully downloaded since the
      last call to update. """
    downloaded = 0

    # Read from the images queue.
    while not _image_queue.empty():
      synset, name, url, image, patches = _image_queue.get()

      if image is None:
        # Download failed.
        self.__failures.add((synset, name, url))
        continue

      if not self.__all_patches:
        # Choose a random patch.
        patch = patches[random.randint(0, len(patches) - 1)]
        self.__mem_buffer.add(patch, name, synset)
      else:
        self.__mem_buffer.add_patches(patches, name, synset)

      # Add it to the disk cache.
      self.__disk_cache.add(image, name, synset)

      downloaded += 1

    return downloaded

  def wait_for_downloads(self):
    """ Waits until all the downloads are finished, basically by calling update
    periodically.
    Returns:
      A list of all the failed downloads. """
    while self.update():
      time.sleep(1)

    return self.get_failures()

  def get_failures(self):
    """ Returns: The list of failures up until now. Every call to this function
    will clear the list after returning it. """
    failures = list(self.__failures)
    self.__failures = set([])
    return failures


# These processes do not need any internal state copied, so we create them at
# module load time.

# Queue other processes use to communicate image data.
_image_queue = multiprocessing.Queue()
# Queue we use to send commands to other processes.
_command_queue = multiprocessing.Queue()

for i in range(0, WORKER_LIMIT):
  worker = DownloaderProcess(_command_queue, _image_queue)
  worker.start()
