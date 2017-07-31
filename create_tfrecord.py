import random
import os
import tensorflow as tf
from dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset

#================== DEFINING THE ARGUMENTS ================#

flags= tf.app.flags


#Dataset directory
# dataset_dir= "/Users/charujaiswal/Downloads/flowers" #NOTE:  Your dataset_dir should be /path/to/flowers and not /path/to/flowers/flowers_photos
#i.e. put the folder in another folder
dataset_dir= "/Users/charujaiswal/PycharmProjects/models/slim/DATASET/flowers"

flags.DEFINE_string('dataset_dir', dataset_dir, 'dataset_dir')

#Validation proportion: proportion of dataset to be used for evaluation
flags.DEFINE_float('validation_size', 0.3,'size of valid')

#Number of shards to split the dataset into
flags.DEFINE_integer('num_shards', 2, 'Int: num of shards to split the TFRecord files into')

#Seed for repeatability
flags.DEFINE_integer('random_seed',0,'random seed for repeatability')

#Output filename for naming the tfrecord file
flags.DEFINE_string('tfrecord_filename', 'flowers_tfrecord', 'Output filename for tfrecord file')

FLAGS= flags.FLAGS

photo_filenames, class_names= _get_filenames_and_classes(FLAGS.dataset_dir)
print class_names

# Refer each of the class names to a specific integer number for predictions later
class_names_to_ids = dict(zip(class_names, range(len(class_names))))
print class_names_to_ids

# Find the number of validation examples we need
num_validation = int(FLAGS.validation_size * len(photo_filenames))

# Divide the training datasets into train and test:
random.seed(FLAGS.random_seed)
random.shuffle(photo_filenames)
training_filenames = photo_filenames[num_validation:]
validation_filenames = photo_filenames[:num_validation]

clean_training_filenames= [value for value in training_filenames if '.DS_Store' not in value ]
clean_validation_filenames= [value for value in validation_filenames if '.DS_Store' not in value ]

# del training_filenames[201]
# del validation_filenames[806] # file 806 is a DS_store non jpeg file '/Users/charujaiswal/Downloads/flowers/flower_photos/tulips/.DS_Store'
# # print validation_filenames[805:810]
# print training_filenames[2168:2171]

# First, convert the training and validation sets.
_convert_dataset('train', clean_training_filenames, class_names_to_ids,
                 dataset_dir=FLAGS.dataset_dir, tfrecord_filename=FLAGS.tfrecord_filename, _NUM_SHARDS=FLAGS.num_shards)
_convert_dataset('validation', clean_validation_filenames, class_names_to_ids,
                 dataset_dir=FLAGS.dataset_dir, tfrecord_filename=FLAGS.tfrecord_filename, _NUM_SHARDS=FLAGS.num_shards)

# Finally, write the labels file:
labels_to_class_names = dict(zip(range(len(class_names)), class_names))
write_label_file(labels_to_class_names, FLAGS.dataset_dir)

print '\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename)