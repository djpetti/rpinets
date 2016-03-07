#!/usr/bin/python


import json
import subprocess
import threading
import time


class Task(threading.Thread):
  """ Represents a single task. It executes in its own thread. """
  class Type(object):
    """ The type of the task. """
    # Task that mostly requires the network.
    NETWORK = "network"
    # Task that mostly requires the CPU.
    CPU = "cpu"
    # Task that mostly requires the GPU.
    GPU = "gpu"

  def __init__(self, task_type, command, suite, print_mutex):
    """
    Args:
      task_type: The type of the task. See above.
      command: The command to actually run for this task.
      suite: The name of the suite that this task belongs to.
      print_mutex: Mutex to grab when we're printing so it doesn't get too
      messy.
    """
    super(Task, self).__init__()

    self.__type = task_type
    self.__command = command

    self.__results = None
    self.__mutex = print_mutex

    self.__suite = suite

  def run(self):
    """ Actually runs the command. """
    self.__process = subprocess.Popen(self.__command, stdout=subprocess.PIPE)
    lines_iterator = iter(self.__process.stdout.readline, b"")
    for line in lines_iterator:
      # If we have results, we should save them so the main process can read
      # them later.
      if line.startswith("results="):
        self.__results = json.loads(line[8:])
        continue

      # Display the line.
      self.__mutex.acquire()
      print line.rstrip("\n") + "\r"
      self.__mutex.release()

  def get_type(self):
    """ Returns: The task type. """
    return self.__type

  def get_results(self):
    """ Gets the results of a test. Meant to be called after the task has
    finished.
    Returns: The saved results. """
    return self.__results

  def get_suite(self):
    """ Returns: The name of the suite it belongs to. """
    return self.__suite

  def get_return_code(self):
    """ Gets the return code of the process. """
    return self.__process.wait()

class Scheduler(object):
  """ Takes a bunch of tasks and figures out how to run them all. """

  def __init__(self, tasks):
    """
    Args:
      tasks: Should be a tuple with a list of the tasks for each individual test
      suite that need to be done. """
    # Copy it so we don't modify the version we passed in.
    self.__tasks = []
    for task in tasks:
      self.__tasks.append(task[:])
    self.__running_tasks = []

  def __get_next_task(self):
    """ Decides which task to start next.
    Returns:
      The task to start next, or None if no new tasks can be started. If there
      are no more tasks to run at all, it returns 0. """
    # Get the next task for each suite.
    next_tasks = []
    has_task = False
    for suite in self.__tasks:
      if suite:
        next_tasks.append(suite[0])
        has_task = True
      else:
        next_tasks.append(None)
    if not has_task:
      # No more tasks to run.
      return 0

    # Decide whether any new ones can be started.
    runnable = []
    first_task_index = None
    for i in range(0, len(next_tasks)):
      task = next_tasks[i]
      if not task:
        continue

      # We have to make sure it doesn't use the same resource as a task that's
      # already running.
      for running_task in self.__running_tasks:
        if running_task.get_type() == task.get_type():
          break
        if running_task.get_suite() == task.get_suite():
          break
      else:
        runnable.append(task)
        if first_task_index == None:
          first_task_index = i

    if not runnable:
      return None
    # Remove the proper one.
    self.__tasks[first_task_index].pop(0)
    return runnable[0]

  def run(self):
    """ Runs all the tasks. """
    while True:
      task = self.__get_next_task()
      if (task == 0 and not self.__running_tasks):
        print "All tasks done."
        return

      if task:
        # We have a new task to start.
        task.start()
        self.__running_tasks.append(task)
        # is_alive() will return False immediately here, so we want to skip all
        # those checks.
        time.sleep(1)
        continue

      # Check if any tasks completed.
      to_delete = []
      for task in self.__running_tasks:
        if not task.is_alive():
          if task.get_return_code():
            # It failed somehow.
            raise RuntimeError("Task failed!")

          to_delete.append(task)
      for task in to_delete:
        self.__running_tasks.remove(task)

      time.sleep(1)


def main():
  # These are the tasks we need to run.
  print_mutex = threading.Lock()

  tensorflow = [ \
      #Task(Task.Type.NETWORK, ["docker", "pull", "djpetti/tensorflow"],
      #     "tensorflow", print_mutex),
      Task(Task.Type.GPU, ["tensorflow/run_tensorflow_tests.sh"],
           "tensorflow", print_mutex)
  ]
  caffe = [ \
      #Task(Task.Type.NETWORK, ["docker", "pull", "djpetti/caffe"],
      #     "caffe", print_mutex),
      Task(Task.Type.GPU, ["caffe/run_caffe_tests.sh"],
           "caffe", print_mutex)
  ]

  scheduler = Scheduler([caffe, tensorflow])
  scheduler.run()

  # Check all the results.
  tensorflow_results = tensorflow[0].get_results()
  caffe_results = caffe[0].get_results()
  print "Tensorflow:"
  print "  time: %f, speed: %f" % (tensorflow_results["mnist"]["elapsed"],
                                   tensorflow_results["mnist"]["speed"])
  print "Caffe:"
  print "  time: %f, speed: %f" % (caffe_results["mnist"]["elapsed"],
                                   caffe_results["mnist"]["speed"])

if __name__ == "__main__":
  main()
