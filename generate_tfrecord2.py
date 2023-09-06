#based on https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd

from tensorflow.python.framework.versions import VERSION
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS


''' 
*************************************************************************
Make sure to edit this method to match the labels you made with labelImg!
*************************************************************************
'''
def class_text_to_int(row_label):
    if row_label == 'cardboard':
        return 1
    elif row_label == 'glass':
        return 2
    elif row_label == 'metal':
        return 3
    elif row_label == 'paper':
        return 4
    elif row_label == 'plastic':
        return 5
    else:
        return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, unique_id):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    image_id = unique_id.get_image_id()

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        # 'image/source_id': dataset_util.bytes_feature(image_id),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

class UniqueId(object):
  """Class to get the unique {image/ann}_id each time calling the functions."""

  def __init__(self):
    self.image_id = 0
    self.ann_id = 0

  def get_image_id(self):
    self.image_id += 1
    return self.image_id

  def get_ann_id(self):
    self.ann_id += 1
    return self.ann_id
  
def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    unique_id = UniqueId()

    for group in grouped:
        tf_example = create_tf_example(group, path, unique_id)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()

# commands:
# python generate_tfrecord2.py --csv_input=../../datasets/trash-dataset-resized/test_labels.csv --image_dir=../../datasets/trash-dataset-resized/test --output_path=../../datasets/trash-dataset-resized/test.record
# python generate_tfrecord2.py --csv_input=../../datasets/trash-dataset-resized/train_labels.csv --image_dir=../../datasets/trash-dataset-resized/train --output_path=../../datasets/trash-dataset-resized/train.record

# python generate_tfrecord2.py --csv_input=../../datasets/microcontroller-detection/test_labels.csv --image_dir=../../datasets/microcontroller-detection/test --output_path=../../datasets/microcontroller-detection/test.record
# python generate_tfrecord2.py --csv_input=../../datasets/microcontroller-detection/train_labels.csv --image_dir=../../datasets/microcontroller-detection/train --output_path=../../datasets/microcontroller-detection/train.record