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

ilr = 0.001
decay_steps = 2500
decay_factor = 0.96

nw = network()


with tf.Session() as sess:
    global_step = tf.Variable(0, trainable=False, name='global_step')
    isTrain = tf.placeholder(tf.bool, shape=())
    
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=tf.get_default_graph())
    train_data, val_data = format_train_data(EPOCHS, BATCH_SIZE)
    iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                               train_data.output_shapes)
    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(train_data)
    validation_init_op = iterator.make_initializer(val_data)

    logits = nw.inference(next_element[0], isTrain)
    loss = nw.loss(logits, next_element[1])


    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(ilr,
                                    global_step,
                                    decay_steps,
                                    decay_factor,
                                    staircase=True)
    lr_summary = tf.summary.scalar('learning_rate', lr)



    optim = tf.train.AdamOptimizer(learning_rate=0.001)
    #optim = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=1e-6)
    train_op = optim.minimize(loss, global_step=global_step)
    
    prediction = tf.argmax(logits, 1)
    equality = tf.equal(prediction, tf.argmax(next_element[1], 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    
    image_summary = tf.summary.image('processed', next_element[0])
      
    
    loss_summary = tf.summary.scalar("loss", loss)
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    
    
    
    val_accuracy = tf.Variable(0.,name='val_accuracy')
    validation_summary = tf.summary.scalar("val_accuracy", val_accuracy)
    
    
    merged_summary_op = tf.summary.merge([loss_summary, accuracy_summary, image_summary])
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=tf.get_default_graph())

    
    
    
    sess.run(tf.global_variables_initializer())
    sess.run(training_init_op)
    saver = tf.train.Saver()
    for epoch in range(EPOCHS):
        sess.run(training_init_op)
        while True:
            try:
                l, a, t, s = sess.run([loss, accuracy, train_op, merged_summary_op], feed_dict={isTrain:True})
                summary_writer.add_summary(s, global_step=tf.train.global_step(sess, global_step))
                print('test H, epoch' , epoch, 'loss', l, 'accuracy', a, 'step', tf.train.global_step(sess, global_step), (tf.train.global_step(sess, global_step)*BATCH_SIZE)%45000, "/45000")
            except tf.errors.OutOfRangeError:
                break
        print('saving model ')
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path)
        sess.run(validation_init_op)
        print('running validation')
        avg_a = []
        while True:
            try:
                a = sess.run(accuracy, feed_dict={isTrain:False})
                avg_a.append(a)
            except tf.errors.OutOfRangeError:
                break
        val_accuracy = tf.assign(val_accuracy, tf.reduce_mean(tf.stack(avg_a)))
        vs, va = sess.run([validation_summary, val_accuracy])
        print('validation accuracy', va)
        summary_writer.add_summary(vs, global_step=tf.train.global_step(sess, global_step))



