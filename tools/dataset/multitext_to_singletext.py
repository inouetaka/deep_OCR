"""Extract Singletexts from Multitext h5 file"""
import math
import os
import cv2
import h5py
import numpy as np

def resize_pad_image(image, bbs, width, height, pad_color):
  """Resize and pad image to (width, height) using pad_color"""
  image, bbs = resize_image(image, bbs, width, height)
  image, bbs = pad_image(image, bbs, width, height, pad_color)
  return image, bbs

def pad_image(image, bbs, expected_width, expected_height, pad_color):
  """Pad image to (expected_width, expected_height) using pad_color"""
  height, width, _ = image.shape
  padded_image = np.ndarray((expected_height, expected_width, pad_color.shape[0]))
  padded_image[:, :] = pad_color
  if expected_width > width:
    index = math.floor((expected_width - width)/2)
    padded_image[:, index:(index+width), :] = image
    bbs[0, :, :] += index
  elif expected_height > height:
    index = math.floor((expected_height - height)/2)
    padded_image[index:(index+height), :, :] = image  # center pad
    bbs[1, :, :] += index
  else:
    padded_image = image
  return padded_image, bbs

def resize_image(image, bbs, expected_width, expected_height):
  """Resize image to (width, height) preserving aspect ratio"""
  height, width, _ = image.shape
  ratio_w = expected_width / width
  ratio_h = expected_height / height
  if ratio_h > ratio_w:
    resized_w, resized_h, ratio = expected_width, round(height * ratio_w), ratio_w
  else:
    resized_w, resized_h, ratio = round(width * ratio_h), expected_height, ratio_h

  resized_image = cv2.resize(image, (resized_w, resized_h), cv2.INTER_CUBIC)
  resized_bbs = bbs * ratio
  return resized_image, resized_bbs

X_INTERVAL = 10
Y_THRESHOLD = 0.9

  # [(min_point, max_point, length, bbs, chars)]
def merge_rects(attrs):
  """Merge rects"""
  results = []
  for bbs, char in sorted(attrs, key=lambda x: np.min(x[0][0, :])):
    c_min = np.amin(bbs, axis=1)
    c_max = np.amax(bbs, axis=1)
    for ret in results:
      r_min, r_max, r_len, r_bbs, r_chars = ret
      if r_len >= 7: # more than 7 letters
        continue
      if r_max[0] + X_INTERVAL < c_min[0]: # too far
        continue
      if r_min[1] <= c_min[1] and r_max[0] >= c_max[0] and r_max[1] >= c_max[1]: # include
        print(f'break: {r_min, r_max, r_len, r_bbs, r_chars} {bbs, char.decode("utf8")}')
        break
      y_overlap = min(r_max[1], c_max[1]) - max(r_min[1], c_min[1])
      if y_overlap < Y_THRESHOLD * min(c_max[1] - c_min[1], r_max[1] - r_min[1]): # not horizontal
        continue
      ret[0] = np.amin([r_min, c_min], axis=0)
      ret[1] = np.amax([r_max, c_max], axis=0)
      ret[2] = ret[2] + 1
      ret[3].append(bbs)
      ret[4].append(char)
      break
    else:
      results.append([c_min, c_max, 1, [bbs], [char]])
  return results

def crop_merged_rects(image, attrs, width, height):
  """Crop images of merged rects"""
  results = []
  average_color = image.mean(axis=(0, 1))
  for min_bbs, max_bbs, _, bbs, chars in merge_rects(attrs):
    max_bbs = np.ceil(max_bbs).astype(np.int)
    min_bbs = np.floor(min_bbs).astype(np.int)
    bbs = np.array(bbs).transpose(1, 2, 0)
    cropped = image[min_bbs[1]:max_bbs[1], min_bbs[0]:max_bbs[0]]
    relative_bbs = bbs - min_bbs[:, None, None]
    cropped, relative_bbs = resize_pad_image(cropped, relative_bbs, width, height, average_color)
    results.append((cropped, relative_bbs, chars))
  return results

def create_out_db(output_dir_path, db_num):
  """Create output db"""
  output_path = os.path.join(output_dir_path, f'SynthText_{db_num:02}.h5')
  print(f'create new db: {output_path}')
  out_db = h5py.File(output_path, 'w')
  out_db.create_group('/data')
  return out_db

NUMBER_PER_DB = int(1e5)

def extracts(input_path, output_path, width, height):
  """Extract singletexts from multitexts"""
  in_db = h5py.File(input_path, 'r')['data']
  out_db_num = 1
  out_db = create_out_db(output_path, out_db_num)
  out_idx = 0
  for key, data in in_db.items():
    print(f'extracting {key}...')
    image = data[:]
    attrs = data.attrs
    flat_attrs = []
    for i in range(attrs['count']):
      bbs = attrs[f'charBB{i}'].transpose(2, 0, 1)
      flat_attrs.extend(zip(bbs, attrs[f'txt{i}']))
    idx = 0
    for cropped, bbs, text in crop_merged_rects(image, flat_attrs, width, height):
      name = f'{key}_{idx:04}'
      out_db['data'].create_dataset(name, data=cropped)
      out_db['data'][name].attrs['charBB'] = bbs
      out_db['data'][name].attrs['txt'] = text
      out_db['data'][name].attrs['label'] = attrs['label']
      idx += 1
      out_idx += 1
      if out_idx > NUMBER_PER_DB * out_db_num:
        out_db.close()
        out_db_num += 1
        out_db = create_out_db(output_path, out_db_num)
  out_db.close()

def main():
  """main function"""
  arg_parser = argparse.ArgumentParser(description='Extract singletexts from multitexts.')
  arg_parser.add_argument('--input-path', required=True, help='Path of multitext SynthText.h5')
  arg_parser.add_argument('--output-path', required=True, help='Path of output directory')
  arg_parser.add_argument(
    '--width', type=int, default=100, help='Widths of images in output singletext h5 file'
  )
  arg_parser.add_argument(
    '--height', type=int, default=32, help='Heights of images in output singletext h5 file'
  )
  parsed_args = arg_parser.parse_args()
  os.makedirs(parsed_args.output_path, exist_ok=True)
  extracts(parsed_args.input_path, parsed_args.output_path, parsed_args.width, parsed_args.height)

if __name__ == '__main__':
  import argparse
  main()
