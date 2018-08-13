#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:48:59 2018

@author: chenzhm
"""

import numpy as np
import tensorflow as tf
import cv2
import time
import os
from utils import resize_image, load_step1, load_step2, step1_result_process
from utils import step2_result_nms, draw_image

box_pb_path='/media/commaai02/disk_1TB/huapu600/box_mark_s1/box_model/faster_inception.pb-30000'
cls_pb_path='/media/commaai02/disk_1TB/huapu600/mei_yi_zhan/funturn_model/vgg16.pb-150005'
sku_classes = np.loadtxt('/media/commaai02/disk_1TB/huapu600/class594.txt', dtype=str).tolist()


def main():
    g1=tf.Graph()
    g2=tf.Graph()
    config=tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess1 = tf.Session(config=config, graph=g1)
    sess2 = tf.Session(config=config, graph=g2)
    
    step1_inputs,step1_boxes,step1_classes,step1_scores = load_step1(g1, box_pb_path,sess1)
    step2_inputs,step2_scores = load_step2(g2, cls_pb_path,sess2)
    
    cap = cv2.VideoCapture(-1)
    cap.set(3,800)
    cap.set(4,600)
    imagedir_for_test = '/media/commaai02/disk_1TB/huapu600/mei_yi_zhan/image'
    imagelist = os.listdir(imagedir_for_test)
    np.random.shuffle(imagelist)
    for step in range(len(imagelist)):
    #while(1):
        detect_dict = {}
        st = time.time()

        detect_dict['image_bgr'] = cv2.imread(os.path.join(imagedir_for_test, imagelist[step]))
        #_, detect_dict['image_bgr'] = cap.read()
        detect_dict['image_rgb'] = cv2.cvtColor(detect_dict['image_bgr'],cv2.COLOR_BGR2RGB)
        feed_dict1={step1_inputs: np.expand_dims(detect_dict['image_rgb'], axis=0)}
        detect_dict['step1_boxes'], detect_dict['step1_classes'], detect_dict['step1_scores'] = sess1.run([step1_boxes,step1_classes,step1_scores], feed_dict=feed_dict1)
        
        detect_dict = step1_result_process(detect_dict, thr=0.8)
        num_step1_detect_boxes = len(detect_dict['step1_classes'])
        step2_scores_list = []
        step2_classes_list = []
        step2_skus_list = []
        for i in range(num_step1_detect_boxes):
            box_i = detect_dict['step1_boxes'][i]
            _image = detect_dict['image_rgb'][int(box_i[0]):int(box_i[2]),int(box_i[1]):int(box_i[3]),:]
            pred = sess2.run([step2_scores],feed_dict={step2_inputs: resize_image(_image)})
            class_index = np.argmax(pred[0],1)[0]
            max_scores = pred[0][0][class_index]
            step2_classes_list.append(class_index)
            step2_scores_list.append(max_scores)
            step2_skus_list.append(sku_classes[class_index])
        detect_dict['step2_scores'] = np.array(step2_scores_list, dtype=np.float32)
        detect_dict['step2_classes'] = np.array(step2_classes_list, dtype=np.int32)
        detect_dict['step2_skus'] = np.array(step2_skus_list, dtype=np.int32)
        
        detect_dict = step2_result_nms(detect_dict, max_ovr=0.5)
        detect_dict = draw_image(detect_dict)
        #cv2.imshow('detector',detect_dict['image_bgr'])
        cv2.imwrite('/media/commaai02/disk_1TB/huapu600/mei_yi_zhan/test/p1/'+imagelist[step], detect_dict['image_bgr'])
        print('time: %f'%(time.time()-st))
        k = cv2.waitKey(10)&0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    main()
    

