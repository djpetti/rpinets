#!/usr/bin/python

import argparse
import os
import cPickle as pickle
import sys

import cache


def _shift_file_left(shift_file, start_offset, length):
  """ Shifts the contents of an open file toward the beginning of that file.
  Args:
    start_offset: The offset in the file to start shifting at.
    length: How many bytes to shift everything after start_offset. """
  # Get size.
  shift_file.seek(0, 2)
  file_size = shift_file.tell()

  # Shift everything in chunks.
  chunk_size = 1000000000
  while start_offset < file_size:
    shift_file.seek(start_offset)
    chunk = shift_file.read(chunk_size)

    # Move it backwards.
    shift_file.seek(start_offset - length)
    shift_file.write(chunk)

    start_offset += len(chunk)

def repair_cache(cache_map, cache_data):
  """ Actually performs the repair of a particular cache.
  Args:
    cache_map: The location of the cache_map file.
    cache_data: The location of the cache_data file. """
  print "Loading cache map..."
  map_file = file(cache_map, "rb")
  map_data, offsets, free_start, free_end = pickle.load(map_file)

  # Go through each entry in the cache one by one. No data should be unnacounted
  # for.
  print "Analyzing cache..."
  data_file = file(cache_data, "r+b")

  # Start with the first element.
  sorted_offsets = sorted(offsets.keys())
  first_img_id = offsets[sorted_offsets[0]]
  label, name = first_img_id.split("_")
  offset, size = map_data[label][name]

  # Go through the entire thing, hopping from one element to the next.
  total_size = os.path.getsize(cache_data)
  while offset <= total_size:
    offset += size

    if offset == free_start:
      # We're at the free section. Skip to the end.
      offset = free_end
    if (offset > free_start and offset < free_end):
      # We somehow ended up in the middle of the free section.
      print "ERROR: Cache data extends into free section."
      # Fix it.
      free_start = offset
      offset = free_end
      print "(Repaired.)"

    # There should be another image here.
    if offset not in offsets:
      print "ERROR: Unexpected empty space in cache."
      space_start = offset
      # Scan forward until we find the end of the empty region.
      while offset not in offsets:
        if offset >= total_size:
          # We read off the end of the cache, in which case we can just cut the
          # file here.
          data_file.seek(space_start)
          data_file.truncate()
          break

        offset += 1
      space_end = offset

      # Move everything to cover the empty region.
      _shift_file_left(data_file, space_end, space_end - space_start)
      print "(Repaired.)"

    # Go to the next image.
    img_id = offsets[offset]
    label, name = img_id.split("_")
    offset, size = map_data[label][name]

  # Write back to files.
  data_file.close()

  map_file = file(cache_map, "wb")
  pickle.dump((map_data, offsets, free_start, free_end), map_file)
  map_file.close()

def main():
  parser = argparse.ArgumentParser(description="Scan cache for errors and \
                                                repair them.")
  parser.add_argument("cache_map", help="The cache map file.")
  parser.add_argument("cache_data", help="The cache data file.")
  args = parser.parse_args()

  if not os.path.exists(args.cache_map):
    print "My dear lad, I'm afraid '%s' does not exist." % (args.cache_map)
    sys.exit(2)
  if not os.path.exists(args.cache_data):
    print "My dear lad, I'm afraid '%s' does not exist." % (args.cache_data)
    sys.exit(2)

  repair_cache(args.cache_map, args.cache_data)


if __name__ == "__main__":
  main()
