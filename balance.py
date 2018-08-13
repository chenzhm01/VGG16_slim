#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 18:58:49 2018

@author: commaai
"""

import numpy as np
import os

skus = np.loadtxt('/media/chenzhm/Data/huapu_bread/class.txt',dtype=str)
image_top_dir = '/media/chenzhm/Data/huapu_bread/crop_images'
sample_num = 384
num_classes = len(skus)

outputdir = '/media/chenzhm/Data/huapu_bread'

def balance_sku():
    collaction_image = []
    collaction_label = []
    
    for i in range(len(skus)):
        impath = []
        skudir = os.path.join(image_top_dir, skus[i])
        if not os.path.exists(skudir):
            continue
        one_sku_image = [os.path.join(skudir, image_i) for image_i in os.listdir(skudir)]
        np.random.shuffle(one_sku_image)
        if len(one_sku_image)==sample_num:
            impath += one_sku_image
        elif len(one_sku_image)<sample_num:
            k = int(sample_num/len(one_sku_image))
            impath += one_sku_image*k
            rk = sample_num-k*len(one_sku_image)
            impath+=one_sku_image[:rk]
        else:
            np.random.shuffle(one_sku_image)
            impath += one_sku_image[:sample_num]
        collaction_image += impath
        collaction_label += [str(i)]*sample_num

    f1 = open(outputdir+'/image.txt','w')
    f2 = open(outputdir+'/label.txt','w')
    _=f1.write(image_top_dir+'/9999999/9999999_9999999_9999999_9999999_9999999_9999999_999999_9999999.jpg\n')
    _=f2.write('9999\n')
    random_ind = np.arange(0, len(collaction_image), dtype=np.int32)
    np.random.shuffle(random_ind)
    for j in range(len(collaction_image)):
        _=f1.write(collaction_image[random_ind[j]]+'\n')
        _=f2.write(collaction_label[random_ind[j]]+'\n')        

    f1.close()
    f2.close()

if __name__ == '__main__':
    balance_sku()
         
