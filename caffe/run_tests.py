#!/usr/bin/python

import subprocess
import time

def _run_with_output(command, cwd="caffe/"):
  """ Runs a comand, and displays the output as it is running.
  Args:
    command: The command to run.
    cwd: Directory to run command in.
  Returns:
    The command output. """
  popen = subprocess.Popen(command, stdout=subprocess.PIPE, cwd=cwd)
  lines_iterator = iter(popen.stdout.readline, b"")

  output = ""
  for line in lines_iterator:
    print(line)
    output += line
  return output

def run_mnist_test():
  """ Basically a wrapper around caffe that goes and trains an MNIST classifier.
  Returns:
    A tuple containing the total elapsed time, and the average number of
    training iterations per second. """
  print("Caffe: Starting MNIST test...")

  train_iterations = 2000

  start_time = time.time()
  _run_with_output(["examples/mnist/train_lenet_rmsprop.sh"])
  elapsed = time.time() - start_time

  # Find how much time it took.
  speed = train_iterations / elapsed
  print("Caffe: Ran %d training iterations. (%f iter/s)" % \
      (train_iterations, speed))
  print("Caffe: MNIST test completed in %f seconds." % (elapsed))
  return (elapsed, speed)

def main():
  run_mnist_test()

if __name__ == "__main__":
  main()
