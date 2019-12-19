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



class network():
    def __init__(self):
        self._mode = None
        
    def inference(self, inputs, isTrain):
        nn = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
        nn = tf.layers.conv2d(inputs=nn, filters=32, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
        nn = tf.layers.max_pooling2d(inputs=nn, pool_size=[2, 2], strides=2)
        nn = tf.layers.dropout(inputs=nn, rate=0.4, training=isTrain)

#        nn = tf.nn.lrn(nn, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        
        nn = tf.layers.conv2d(inputs=nn, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
        nn = tf.layers.conv2d(inputs=nn, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
        nn = tf.layers.max_pooling2d(inputs=nn, pool_size=[2, 2], strides=2)
        nn = tf.layers.dropout(inputs=nn, rate=0.4, training=isTrain)
        
#        nn = tf.nn.lrn(nn, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        
        #16,16,64
        nn = tf.reshape(nn,[-1, 8*8*64])
        nn = tf.layers.dense(nn, 512, activation=tf.nn.relu)
        nn = tf.layers.dropout(inputs=nn, rate=0.4, training=isTrain)
        nn = tf.layers.dense(nn, units=10, name="softmax_tensor")
        return nn
        
        

    def loss(self, logits, labels):
        self.cost = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        #self.cost += tf.losses.get_regularization_loss()
        return self.cost 