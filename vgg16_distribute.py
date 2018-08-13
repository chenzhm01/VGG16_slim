#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:47:31 2018

@author: commaai02
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from deployment import model_deploy

import numpy as np
from nets import vgg
from nets import resnet_v2
from nets import inception_v1
from tensorflow.python.framework import graph_util
import os
import utils

slim = tf.contrib.slim

fine_tune_path = '/media/chenzhm/Data/ImageNet_model/vgg_16.ckpt'
output_path = '/media/chenzhm/Data/ImageNet_model/output_multi_worker'

batch_size = 2
num_classes = 47
max_steps = 6016

sync_replicas = True

VGG_MEAN_rgb = [123.68, 116.779, 103.939]

def reader():
    all_images_path = np.loadtxt('/media/chenzhm/Data/huapu_bread/image.txt',dtype=np.str)[1:]
    all_labels = np.loadtxt('/media/chenzhm/Data/huapu_bread/label.txt', dtype=np.int32)[1:]
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
    net = slim.conv2d(net, num_classes, [1, 1],activation_fn=None,normalizer_fn=None,scope='xfc8')
    net = tf.squeeze(net, axis=[1,2])
    return net

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=1,
        clone_on_cpu=True,
        replica_id=0,
        num_replicas=2,
        num_ps_tasks=1)
    
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    images, labels = reader()
    with tf.device(deploy_config.inputs_device()):
      #images, labels = reader()
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config)

    def clone_fn(batch_queue):
      images, labels = batch_queue.dequeue()
      logits = models(images)
      slim.losses.softmax_cross_entropy(logits, labels)
      return logits

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    with tf.device(deploy_config.optimizer_device()):
      learning_rate = tf.train.piecewise_constant(global_step, [10000, 12000], [0.001, 0.0001, 0.00001])
      optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))
    
    if sync_replicas:
      optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=2,
          total_num_replicas=2,
          variable_averages=None,
          variables_to_average=None)

    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=tf.trainable_variables())
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    
    variables_to_restore = slim.get_variables_to_restore(exclude=['global_step'])
    init_fn = slim.assign_from_checkpoint_fn(fine_tune_path, variables_to_restore, ignore_missing_vars=True)
    print('start~~~~~~~~~~~~')
    session_config=tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    slim.learning.train(
        train_tensor,
        logdir=output_path,
        master='grpc://192.168.10.47:2222',
        is_chief=True,
        init_fn=init_fn,
        summary_op=summary_op,
        number_of_steps=max_steps,
        log_every_n_steps=1,
        save_summaries_secs=60,
        save_interval_secs=600,
        sync_optimizer=optimizer,
        session_config=session_config)

if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES']='0'
  tf.app.run()
