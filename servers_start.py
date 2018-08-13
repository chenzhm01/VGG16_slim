#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:39:27 2018

@author: commaai02
"""
#python servers_start.py --job_name=ps --task_index=0 --gpu=0
#python servers_start.py --job_name=worker --task_index=0 --gpu=0
#python servers_start.py --job_name=worker --task_index=1 --gpu=1


import tensorflow as tf
import os

tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string("gpu", "", "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

def main(_):
    ps_hosts = ['192.168.10.119:2221']
    worker_hosts = ['192.168.10.47:2222', '192.168.10.47:2223']
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    server.join()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu
    tf.app.run()
