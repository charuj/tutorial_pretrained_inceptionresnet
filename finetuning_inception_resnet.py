
import os
import tensorflow as tf
import inception_resnet_v2  #From: https://github.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py
import inception_preprocessing # From: https://github.com/tensorflow/models/blob/master/slim/preprocessing/inception_preprocessing.py
from tensorflow.contrib import slim

###### State where the relevant directories (incl. checkpoint) are located

#Dataset directory where the tfrecord files are located
dataset_dir = "/Users/charujaiswal/PycharmProjects/models/slim/DATASET/flowers"

#State where your log file is at. If it doesn't exist yet, it will be created below in a function.
log_dir = './log'

#Location of checkpoint file (see hackaday.io as mentioned aboved for how to structure directory)
checkpoint_file = "/Users/charujaiswal/PycharmProjects/models/slim/CHECKPOINTDIR/inception_resnet_v2_2016_08_30.ckpt"


#The image size you're resizing your images to-- here  we use the default inception size of 299.
image_size = inception_resnet_v2.inception_resnet_v2.default_image_size # 299

#Number of classes to predict:
num_classes = 5 # the flowers dataset has 5 classes

#Batch size
batch_size= 10

#Number of epochs
num_epochs= 5

#learning rate
lr= 0.01

#Create the file pattern of your TFRecord files so that it could be recognized later on in the get_split function
file_pattern = 'flowers_tfrecord_%s_*.tfrecord'


#State the labels file and then read it
labels_file = '/Users/charujaiswal/PycharmProjects/models/slim/DATASET/flowers/labels.txt'
labels = open(labels_file, 'r')

#Create a dictionary to refer each label to their string name, used in get_split function (required by Dataset class later)
labels_to_name = {}
for line in labels:
    label, string_name = line.split(':')
    string_name = string_name[:-1] #Remove newline
    labels_to_name[int(label)] = string_name

#Create item descriptions, used for the get_split function below. Required by the Dataset class later.
items_to_descriptions = {
    'image': 'A 3-channel RGB coloured flower image, with 5 classes (tulips, sunflowers, roses, dandelion, or daisy).',
    'label': 'Labels of the classes, 0:daisy, 1:dandelion, 2:roses, 3:sunflowers, 4:tulips'
}



# Here we create a function that creates a Dataset class which will give us TFRecord files to feed into a queue in parallel.
def get_split(split_name, dataset_dir, file_pattern=file_pattern):

    """Gets a dataset tuple with instructions for reading the data.
    Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.

    Returns:
    A `Dataset` namedtuple.
    Raises:
    ValueError: if `split_name` is not a valid train/validation split.
    """
    # First check whether the split_name is train or validation
    if split_name not in ['train', 'validation']:
      raise ValueError('The split_name %s is not recognized.' % (split_name))

    # Create a path to locate the tfrecord files, using file_pattern of the name
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    # Create a reader that outputs the records from a tfrecord file
    reader = tf.TFRecordReader

    #Count the number of samples
    num_samples = 0
    file_pattern_for_counting = 'flowers_tfrecord' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if
                          file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    ##Next we create the keys_to_features and item_to_handlers dictionaries and the decoder which are used later by the
    # DatasetDataProvider object to decode tf-example into tensor objects.

    ''' More detail on the DatasetDataProvider later
    '''

    # Create the keys_to_features dictionary for the decoder
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    # Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    # Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    # Create the labels_to_name file
    labels_to_name_dict = labels_to_name

    # Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources=file_pattern_path,
        decoder=decoder,
        reader=reader,
        num_readers=4,
        num_samples=num_samples,
        num_classes=num_classes,
        labels_to_name=labels_to_name_dict,
        items_to_descriptions=items_to_descriptions)

    return dataset


def load_batch(dataset, batch_size=batch_size, height=image_size, width=image_size, is_training=True):
    """Loads a single batch of data for training.

    Args:
      dataset: The dataset to load, created in the get_split function.
      batch_size: The number of images in the batch.
      height(int): int value that is the size the image will be resized to during preprocessing
      width: The size that the image will be resized to during preprocessing
      is_training: Whether or not we're currently training or evaluating.

    Returns:
      images: A Tensor of size [batch_size, height, width, channels(3)], image samples that have been preprocessed, that contain one batch of images.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes (requires one-hot encodings)
    """

    # First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity=32,
        common_queue_min=8)

    # Obtain the raw image using the get method
    raw_image, label = data_provider.get(['image', 'label'])

    # Perform the correct preprocessing for this image depending if it is training or evaluating
    image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)

    # Preprocess the image for display purposes.
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    # Batch up the images by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=4 * batch_size,
        allow_smaller_final_batch=True)

    return images, raw_images, labels



def run():
    # Create the log directory here. When training, it's helpful to train and evaluate progress in real-time.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # ======================= TRAINING =========================
    # Construct the graph and build our model
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)  # Sets the threshold for what messages will be logged, in this case it is set to 'INFO'

        # Get dataset and load a batch
        dataset = get_split('train', dataset_dir, file_pattern=file_pattern)
        images, _, labels = load_batch(dataset, batch_size=batch_size)


        #Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2.inception_resnet_v2(images, num_classes=dataset.num_classes, is_training=True)

        #Scopes that you want to exclude for restoration, from the checkpoint
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        #One-hot encode the labels
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

        #Specify the loss function;
        # slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        # total_loss = slim.losses.get_total_loss()
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
        total_loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # State the metrics that you want to predict.
        predictions = tf.argmax(end_points['Predictions'], 1)
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)

        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=log_dir,
            number_of_steps=num_epochs)


    print('Finished training. Last batch loss %f' % final_loss)



if __name__ == '__main__':
    run()
