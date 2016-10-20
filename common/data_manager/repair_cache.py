#!/usr/bin/python

import argparse
import os
import cPickle as pickle
import sys

import cv2

import numpy as np

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
    print start_offset

def _check_image_loadable(cache_file, offset, size):
  """ Check that we can load a particular image.
  Args:
    cache_file: The file storing the image data.
    offset: The offset in the file where the image starts.
    size: The size of the image.
  Returns:
    True if the image is loadable, False if it isn't. """
  cache_file.seek(offset)
  raw_data = cache_file.read(size)
  raw_data = np.asarray(bytearray(raw_data), dtype=np.uint8)

  image = cv2.imdecode(raw_data, cv2.CV_LOAD_IMAGE_UNCHANGED)
  return image != None

def _remove_offset(offset, offsets, map_data):
  """ Remove a bad offset.
  Args:
    offset: The offset to remove.
    offsets: The dict of offsets.
    map_data: The map data dict. """
  img_id = offsets[offset]
  offsets.pop(offset)
  label, name = img_id.split("_")
  map_data[label].pop(name)

def repair_cache(cache_map, cache_data, check_images):
  """ Actually performs the repair of a particular cache.
  Args:
    cache_map: The location of the cache_map file.
    cache_data: The location of the cache_data file.
    check_images: Whether to check that all images can actually be loaded. """
  print "Loading cache map..."
  map_file = file(cache_map, "rb")
  map_data, offsets, free_start, free_end = pickle.load(map_file)

  # Go through each entry in the cache one by one. No data should be unnacounted
  # for.
  print "Analyzing cache..."
  data_file = file(cache_data, "r+b")

  # If we move stuff in the file, we're going to have to start shifting offsets.
  # This specifies by how much.
  offset_shift = 0

  total_size = os.path.getsize(cache_data)
  sorted_offsets = sorted(offsets.keys())
  # Remove offsets that are off the end of the file.
  for offset in reversed(sorted_offsets):
    if offset >= total_size:
      print "ERROR: Offset %d is out of range." % (offset)
      # Remove it.
      _remove_offset(offset, offsets, map_data)
    else:
      # Everything else is smaller.
      break

  # Start with the first element.
  first_img_id = offsets[sorted_offsets[0]]
  label, name = first_img_id.split("_")
  offset, size = map_data[label][name]

  # Go through the entire thing, hopping from one element to the next.
  file_shifts = []
  last_percentage = 0
  percentage = 0
  while offset < total_size:
    # Check that the image can be loaded.
    if check_images:
      if not _check_image_loadable(data_file, offset, size):
        print "ERROR: Unloadable image at %d." % (offset)
        file_shifts.append((offset + size, size))

        # Remove the offset.
        _remove_offset(offset, offsets, map_data)

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

    # There should be another image here.
    end_of_file = False
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
          end_of_file = True
          print "Hit EOF."
          break

        offset += 1
      space_end = offset

      # Move everything to cover the empty region.
      if not end_of_file:
        file_shifts.append((space_end, space_end - space_start))

    if end_of_file:
      # We hit the end of the file.
      print "EOF, exiting."
      break

    # Go to the next image.
    img_id = offsets[offset]
    label, name = img_id.split("_")
    offset, size = map_data[label][name]

    # Show percentage.
    percentage = float(offset) / total_size * 100
    if percentage - last_percentage > 1:
      print "(%d percent done.)" % (percentage)
      last_percentage = percentage

  # Remove all the gaps in the file.
  offset_shift = 0
  update_offsets = []
  print file_shifts
  for gap_end, size in file_shifts:
    print "Removing gap at %d." % (gap_end)

    # Since we've been pushing stuff back, we'll have to update our numbers
    # here.
    gap_end -= offset_shift

    # Remove the space.
    _shift_file_left(data_file, gap_end, size)
    offset_shift += size

    # Record this information so we can update the offset data structures later.
    update_offsets.append((gap_end - size, offset_shift))

  # Update the actual offsets in the data structure.
  print "Finalizing cache map..."
  label, name = first_img_id.split("_")
  offset, size = map_data[label][name]

  print update_offsets
  if update_offsets:
    gap_index = 0
    next_gap_start, next_shift_after = update_offsets[gap_index]
    updated_free_space = False
    last_offset = None
    last_size = None
    for offset in sorted_offsets:
      if offset not in offsets:
        # This offset was a bad image and we removed it.
        continue

      label, name = offsets[offset].split("_")
      _, size = map_data[label][name]

      # Make sure it doesn't overlap.
      if (last_offset != None and offset < last_offset + last_size):
        print "ERROR: Overlapping offset: %d" % (offset)
        _remove_offset(offset, offsets, map_data)
        continue
      last_offset = offset
      last_size = size

      if offset > next_gap_start:
        if gap_index < len(update_offsets) - 1:
          gap_index += 1
          next_gap_start, next_shift_after = update_offsets[gap_index]

      if gap_index > 0:
        gap_start, shift_after = update_offsets[gap_index - 1]

        # Shift the offsets.
        new_offset = offset - shift_after
        offsets[new_offset] = offsets[offset]
        offsets.pop(offset)

        map_data[label][name] = (new_offset, size)

        # We also need to deal with our free space if we pushed things back.
        if (not updated_free_space and free_start > gap_start):
          free_start -= shift_after
          free_end -= shift_after
          updated_free_space = True

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
  parser.add_argument("-c", "--check-images", action="store_true",
                      help="Check to make sure that every image can be loaded.")
  args = parser.parse_args()

  if not os.path.exists(args.cache_map):
    print "My dear lad, I'm afraid '%s' does not exist." % (args.cache_map)
    sys.exit(2)
  if not os.path.exists(args.cache_data):
    print "My dear lad, I'm afraid '%s' does not exist." % (args.cache_data)
    sys.exit(2)

  repair_cache(args.cache_map, args.cache_data, args.check_images)


if __name__ == "__main__":
  main()
