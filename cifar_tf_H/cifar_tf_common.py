import os
import numpy as np
import pickle
import gzip
import random
import tensorflow as tf
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'output',
                           """Directory where to write event logs """
                           """and checkpoint.""")




EPOCHS = 50
BATCH_SIZE = 32

training_dataset = ['../../data/cifar-10-batches-py/data_batch_1', 
                    '../../data/cifar-10-batches-py/data_batch_2', 
                    '../../data/cifar-10-batches-py/data_batch_3', 
                    '../../data/cifar-10-batches-py/data_batch_4', 
                    '../../data/cifar-10-batches-py/data_batch_5', ]
test_dataset = ['../../data/cifar-10-batches-py/test_batch']






def find_label(item):
    labels = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trunk']
    return labels[item]

def image_generalization(frame):
    frame = tf.image.random_brightness(frame, max_delta=63)
    frame = tf.image.random_contrast(frame, lower=0.2, upper=1.8)
    return tf.image.random_flip_left_right(frame)


def save_images(dataset, n):
    iterator = dataset.make_one_shot_iterator()
    fig = plt.figure(0, figsize=[2, n*2])
    with tf.Session() as sess:
        for frame in range(n):
            a = fig.add_subplot(n,1,frame+1)
            item = iterator.get_next()
            p = sess.run(item)
            plt.imshow(p[0].astype(int))
            label= find_label(np.argmax(p[1]))
            a.set_title(label)
    plt.savefig('dumb.png')
    
    
def format_train_data(epochs, batch_size, split=45000):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    dataset_data = tf.data.Dataset.from_tensor_slices(x_train)
    dataset_data = dataset_data.map(lambda frame: tf.cast(frame, tf.float32))
    #dataset_data = dataset_data.map(lambda frame: tf.reshape(frame,[3,32,32]))
    #dataset_data = dataset_data.map(lambda frame: tf.transpose(frame,[1,2,0]))
    dataset_data = dataset_data.map(lambda frame: image_generalization(frame))
    dataset_data = dataset_data.map(lambda frame: tf.image.per_image_standardization(frame))
    dataset_labels = tf.data.Dataset.from_tensor_slices(y_train)
    dataset_labels = dataset_labels.map(lambda label: tf.one_hot(label[0], 10))
    dataset = tf.data.Dataset.zip((dataset_data, dataset_labels))
    #save_images(dataset,4)

    train_dataset = dataset.take(split)
    #train_dataset = train_dataset.repeat(epochs)
    val_dataset = dataset.skip(split)
    #val_dataset = dataset.take(train_size)
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset
    
def format_test_data(epochs, batch_size, split=45000):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    dataset_visual = tf.data.Dataset.from_tensor_slices(x_test)
    dataset_data = tf.data.Dataset.from_tensor_slices(x_test)
    dataset_data = dataset_data.map(lambda frame: tf.cast(frame, tf.float32))
    dataset_data = dataset_data.map(lambda frame: tf.image.per_image_standardization(frame))
    dataset_labels = tf.data.Dataset.from_tensor_slices(y_test)
    dataset_labels = dataset_labels.map(lambda label: tf.one_hot(label[0], 10))
    testset = tf.data.Dataset.zip((dataset_data, dataset_labels))
    testset = testset.batch(batch_size)

    return testset, dataset_visual


    

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    print(j)
    e = np.zeros([10])
    e[j] = 1.0
    return e
    
