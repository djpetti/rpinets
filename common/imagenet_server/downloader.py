""" Class that downloads and processes requested images in separate processes.
"""


import collections
import logging
import multiprocessing
import random
import signal
import time

import data_augmentation
import images


logger = logging.getLogger(__name__)


class DownloaderProcess(multiprocessing.Process):
  """ A process that downloads a single image and exits. """
  def __init__(self, synset, number, url, image_queue,
               *args, **kwargs):
    """ Args:
      synset: The synset of the image to download.
      number: The number of the image in the synset.
      url: The URL to download the image from.
      image_queue: Queue that we write image information onto. """
    super(DownloaderProcess, self).__init__(*args, **kwargs)

    self.__synset = synset
    self.__number = number
    self.__url = url
    self.__image_queue = image_queue

  def run(self):
    logger.debug("Starting downloader process.")

    # Download the image.
    image = images.download_image(self.__url, keep_color=True)
    if image is None:
      logger.warning("Failed to download %s." % (self.__url))
      self.__image_queue.put((self.__synset, self.__number, self.__url, None,
                              None))
      return

    # We should choose a patch here before it gets put in the buffer.
    patches = data_augmentation.extract_patches(image)

    # Save the image to the queue.
    logger.debug("Saving image: %s_%s" % (self.__synset, self.__number))
    # Defer acting on a terminate signal until after the queue call. (Which can
    # block, but won't indefinitely as long as update() is being called
    # regularly.)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    self.__image_queue.put((self.__synset, self.__number, self.__url, image,
                            patches))

  def start(self, *args, **kwargs):
    self.__start_time = time.time()

    super(DownloaderProcess, self).start(*args, **kwargs)

  def get_start_time(self):
    """ Returns: The time at which this process was started. """
    return self.__start_time

  def get_info(self):
    """ Returns: Image synset, number and url. """
    return self.__synset, self.__number, self.__url


class DownloadManager(object):
  """ Deals with managing and dispatching downloads. """

  def __init__(self, process_limit, disk_cache, mem_buffer,
               patch_separation=-1, timeout=15):
    """
    Args:
      process_limit: Maximum number of downloads we can run at one time.
      disk_cache: DiskCache to save downloaded images to.
      mem_buffer: MemoryBuffer to save downloaded images to.
      patch_separation: How much to separate each patch of the same image in the
      memory buffer. Defaults to -1, in which case only one random patch will be
      saved for each image.
      timeout: How long downloader processes can run before we terminate them. """
    self.__process_limit = process_limit
    self.__disk_cache = disk_cache
    self.__mem_buffer = mem_buffer
    self.__patch_separation = patch_separation
    self.__timeout = timeout

    # Downloads that are waiting to start.
    self.__download_queue = collections.deque()
    # Queue other processes use to communicate image data.
    self.__image_queue = multiprocessing.Queue()
    # Set of failed downloads.
    self.__failures = set([])
    # Set of downloads that were terminated.
    self.__terminated = set([])

  def download_new(self, synset, number, url):
    """ Registers a new image to be downloaded.
    Args:
      synset: The synset of the image.
      number: The image number in the synset.
      url: The url of the image. """
    # Add a new download.
    download_process = DownloaderProcess(synset, number, url,
                                         self.__image_queue)
    self.__download_queue.append(download_process)

  def update(self):
    """ Adds any new processes, and cleans up any old ones. Should be called
    periodically.
    Returns:
      True if there are still more downloads pending, False otherwise. """
    # Check for any download processes that have taken too long.
    for process in multiprocessing.active_children():
      if (time.time() - process.get_start_time()) > self.__timeout:
        logger.warning("Terminating downloader process that took too long.")
        if process.get_info() not in self.__terminated:
          process.terminate()
          # Add it to the failures list, because it counts.
          self.__failures.add(process.get_info())
          self.__terminated.add(process.get_info())

    processes = True
    while len(multiprocessing.active_children()) < self.__process_limit:
      # We're free to add more processes.
      if not len(self.__download_queue):
        if not multiprocessing.active_children():
          # Nothing more to download.
          processes = False
        break

      new_process = self.__download_queue.popleft()
      new_process.start()

    # Read from the images queue.
    data = True
    if self.__image_queue.empty():
      data = False
    while not self.__image_queue.empty():
      synset, name, url, image, patches = self.__image_queue.get()

      if (synset, name, url) in self.__terminated:
        # If it was terminated, this was already reported as a failure.
        continue

      if image is None:
        # Download failed.
        self.__failures.add((synset, name, url))
        continue

      if self.__patch_separation < 0:
        # Choose a random patch.
        patch = patches[random.randint(0, len(patches) - 1)]
        self.__mem_buffer.add(patch, name, synset)
      else:
        self.__mem_buffer.add_patches(self.__patch_separation, patches, name,
                                      synset)

      # Add it to the disk cache.
      self.__disk_cache.add(image, name, synset)

    if not (processes or data):
      # We're done downloading everything, so we can clear the terminated set.
      self.__terminated = set([])
    return processes or data

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