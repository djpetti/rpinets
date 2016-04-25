""" Class that downloads and processes requested images in separate processes.
"""


import collections
import logging
import multiprocessing
import time

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
    if image == None:
      logging.warning("Failed to download %s." % (self.__url))
      self.__image_queue.put((self.__synset, self.__number, self.__url, None))
      return
    image = images.reshape_image(image)

    # Save the image to the queue.
    logging.debug("Saving image: %s_%s" % (self.__synset, self.__number))
    self.__image_queue.put((self.__synset, self.__number, self.__url, image))


class DownloadManager(object):
  """ Deals with managing and dispatching downloads. """

  def __init__(self, process_limit, caches):
    """
    Args:
      process_limit: Maximum number of downloads we can run at one time.
      caches: List of Caches to save images to. """
    self.__process_limit = process_limit
    self.__caches = caches

    # Downloads that are waiting to start.
    self.__download_queue = collections.deque()
    # Queue other processes use to communicate image data.
    self.__image_queue = multiprocessing.Queue()
    # List of failed downloads.
    self.__failures = []

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
      synset, name, url, image = self.__image_queue.get()
      if image == None:
        # Download failed.
        self.__failures.append((synset, name, url))
        continue

      # Otherwise, add it to the caches.
      for cache in self.__caches:
        cache.add(image, name, synset)

    return processes or data

  def wait_for_downloads(self):
    """ Waits until all the downloads are finished, basically by calling update
    periodically.
    Returns:
      A list of all the failed downloads. """
    while self.update():
      time.sleep(1)

    return self.__failures

  def get_failures(self):
    """ Returns: The list of failures up until now. Every call to this function
    will clear the list after returning it. """
    failures = self.__failures[:]
    self.__failures = []
    return failures
