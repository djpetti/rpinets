""" Class that downloads and processes requested images in separate processes.
"""


from Queue import Empty
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
      req_id, synset, number, url, patch_shape = self.__command_queue.get()
      self.__download_image(req_id, synset, number, url, patch_shape)

  def __download_image(self, req_id, synset, number, url, patch_shape):
    """ Downloads a single image.
    Args:
      req_id: The ID of the requesting download manager.
      synset: The image synset.
      number: The image number.
      url: The image url.
      patch_shape: The shape of the patches to extract. If None, it will not
                   extract patches. """
    logger.debug("Downloading image for %d: %s_%s" % (req_id, synset, number))

    # Download the image.
    image = images.download_image(url, keep_color=True)
    if image is None:
      logger.warning("Failed to download %s." % (url))
      self.__image_queue.put((req_id, synset, number, url, None, None))
      return

    # Extract the patches to send back.
    patches = None
    if patch_shape:
      patches = data_augmentation.extract_patches(image, patch_shape)

    # Save the image to the queue.
    logger.debug("Sending image for %d: %s_%s" % (req_id, synset, number))
    self.__image_queue.put((req_id, synset, number, url, image,
                            patches))


class DownloadManager(object):
  """ Deals with managing and dispatching downloads. """

  # Counter that gives each instance a unique id.
  _current_id = 0

  def __init__(self, disk_cache, mem_buffer,
               patch_shape=None, all_patches=False):
    """
    Args:
      disk_cache: DiskCache to save downloaded images to.
      mem_buffer: MemoryBuffer to save downloaded images to.
      patch_shape: The shape that extracted patches should be. It defaults to
                   None, in which case no patches will be extracted at all.
      all_patches: Whether to save all the patches, or just pick one at random.
      """
    self.__id = DownloadManager._current_id
    DownloadManager._current_id += 1

    self.__disk_cache = disk_cache
    self.__mem_buffer = mem_buffer
    self.__patch_shape = patch_shape
    self.__all_patches = all_patches

    # Set of failed downloads.
    self.__failures = set()
    # Set of pending downloads.
    self.__pending = set()

    # Create a new queue for us.
    _rejected_queue[self.__id] = collections.deque()

  def download_new(self, synset, number, url):
    """ Registers a new image to be downloaded.
    Args:
      synset: The synset of the image.
      number: The image number in the synset.
      url: The url of the image.
    Returns:
      True if the download was added, False if it wasn't because it was already
      being downloaded. """
    if (synset, number) in self.__pending:
      logger.debug("Download for %s_%s is already pending." % (synset, number))
      return False

    # Add a new download.
    self.__pending.add((synset, number))
    _command_queue.put((self.__id, synset, number, url, self.__patch_shape))
    return True

  def update(self):
    """ Adds any new processes, and cleans up any old ones. Should be called
    periodically.
    Returns:
      The number of new images that were successfully downloaded since the
      last call to update. """
    downloaded = 0

    # Read from the images queue.
    while True:
      req_id = self.__id
      synset = None
      name = None
      url = None
      image = None
      patches = None

      more_data = False
      if len(_rejected_queue[self.__id]):
        # Get messages that other DownloadManagers read but that are intended
        # for us.
        synset, name, url, image, patches = _rejected_queue[self.__id].pop()
        more_data = True

      else:
        # Read from the main queue.
        try:
          req_id, synset, name, url, image, patches = _image_queue.get_nowait()
          more_data = True
        except Empty:
          # Nothing more to read.
          pass

        if req_id != self.__id:
          # This is for a different DownloadManager.
          logger.debug("%d: Got message for id %d." % (self.__id, req_id))
          _rejected_queue[req_id].appendleft((synset, name, url, image, patches))
          continue

      if not more_data:
        # Nothing more to read.
        break

      self.__pending.remove((synset, name))

      if image is None:
        # Download failed.
        self.__failures.add((synset, name, url))
        continue

      if not self.__patch_shape:
        # No patches. Use the entire image.
        self.__mem_buffer.add(image, name, synset)
      else:
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
_command_queue = multiprocessing.Queue(1500)
# When we read images that belong to the wrong downloader instance, this is
# where they go.
_rejected_queue = {}

for i in range(0, WORKER_LIMIT):
  worker = DownloaderProcess(_command_queue, _image_queue)
  worker.start()
