#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:16:04 2018

@author: commaai02
"""

import tensorflow as tf
import numpy as np
from nets import vgg
from nets import resnet_v2
from tensorflow.python.framework import graph_util
import os
import utils

slim = tf.contrib.slim

fine_tune_path = '/media/commaai02/disk_1TB/ImageNet_model/vgg_16.ckpt'
output_path = '/media/commaai02/disk_1TB/huapu_bread/vgg_16_output'

batch_size = 64
num_classes = 47
max_steps = 6016

VGG_MEAN_rgb = [123.68, 116.779, 103.939]


def reader():
    all_images_path = np.loadtxt('/media/commaai02/disk_1TB/huapu_bread/image.txt',dtype=np.str)[1:]
    all_labels = np.loadtxt('/media/commaai02/disk_1TB/huapu_bread/label.txt', dtype=np.int32)[1:]
    file_dir_queue = tf.train.slice_input_producer([all_images_path, all_labels],shuffle=True,capacity=512)
    img_contents = tf.read_file(file_dir_queue[0])
    label = tf.cast(tf.one_hot(file_dir_queue[1], num_classes), tf.float32)
    image = tf.image.decode_jpeg(img_contents, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.3)
    image = image - VGG_MEAN_rgb
    image = utils.resize_image_2(image, 224)
    image_batch, label_batch = tf.train.batch([image, label], batch_size, capacity=128)
    return image_batch, label_batch

def models(inputs, is_training=True, dropout_keep_prob=0.5):
    net, endpoints = vgg.vgg_16(inputs, num_classes=None, is_training=is_training, dropout_keep_prob=dropout_keep_prob)
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
    net = slim.conv2d(net, num_classes, [1, 1],activation_fn=None,normalizer_fn=None,scope='fc8')
    net = tf.squeeze(net, axis=[1,2])
    return net

def train():
    slim.create_global_step()
    image_batch, label_batch = reader()
    logit = models(image_batch)
    slim.losses.softmax_cross_entropy(logit, label_batch)
    total_loss = slim.losses.get_total_loss(add_regularization_losses=True)
    tf.summary.scalar('loss',total_loss)

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit, axis=1), 
                                          tf.argmax(label_batch, axis=1)), dtype=tf.float32))
    tf.summary.scalar('accuracy', acc)
    
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)
    variables_to_restore = slim.get_variables_to_restore(exclude=['global_step'])
    init_fn = slim.assign_from_checkpoint_fn(fine_tune_path, variables_to_restore, ignore_missing_vars=True)
    slim.learning.train(train_op=train_op,
                        logdir=output_path, 
                        init_fn=init_fn, 
                        number_of_steps=max_steps,
                        save_summaries_secs=200, 
                        save_interval_secs=600)

def frozen():
    ckpt_path = '/media/commaai02/disk_1TB/bread_lxw_zb_0730/vgg_16_output/model.ckpt-6000'
    pb_path = '/media/commaai02/disk_1TB/bread_lxw_zb_0730/vgg_16_output/vgg16.pb-6000'
    x = tf.placeholder(tf.float32, shape=[None, None, 3], name="input_node")
    image = x - VGG_MEAN_rgb
    image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)

    logit = models(tf.expand_dims(image,0),is_training=False,dropout_keep_prob=1.0)
    output_pred = tf.nn.softmax(logit, name="output")

    sess = tf.Session()
    saver = tf.train.Saver()
    initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(initop)
    saver.restore(sess, ckpt_path)
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,["output"])
    with tf.gfile.GFile(pb_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    train()
    #frozen()
    
