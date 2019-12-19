from cifar_tf_adv import network
from cifar_tf_common import *

import os
import numpy as np
import pickle
import gzip
import random
import tensorflow as tf
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt


''' CIFAR10 on TF for practice



'''



nw = network()


with tf.Session() as sess:
    test_data, _ = format_test_data(EPOCHS, BATCH_SIZE, 10000)
    isTrain = tf.placeholder(tf.bool, shape=())
    iterator = test_data.make_one_shot_iterator()
    next_element = iterator.get_next()

    logits = nw.inference(next_element[0], isTrain)
    prediction = tf.argmax(logits, 1)
    equality = tf.equal(prediction, tf.argmax(next_element[1], 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        print('restoring from', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    avg_a = []
    print('running test batch')
    i = 0
    while True:
        try:
            a = sess.run(accuracy, feed_dict={isTrain:False})
            print('test H accuracy', a, 'step', i, i*BATCH_SIZE%10000, "/10000")
            avg_a.append(a)
            i+=1
        except tf.errors.OutOfRangeError:
            break
    print('test accuracy', np.mean(avg_a))



