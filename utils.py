import numpy as np
import tensorflow as tf
import cv2

def resize_image(image):
    h,w,_ = image.shape
    r = 224.0/float(max([h,w]))
    new_shape = [int(w*r),int(h*r)]
    new_shape[np.argmax([w,h])]=224
    image = cv2.resize(image, (new_shape[0], new_shape[1]))
    #b,g,r = cv2.split(image)
    #image = cv2.merge([cv2.equalizeHist(b),cv2.equalizeHist(g),cv2.equalizeHist(r)])
    return image


def resize_image_2(image, max_side=224):
  shape = tf.shape(image)
  height = tf.to_float(shape[0])
  width = tf.to_float(shape[1])
  scale = tf.cond(tf.greater(height, width),
                  lambda: tf.to_float(max_side)/height,
                  lambda: tf.to_float(max_side)/width)
  new_height = tf.to_int32(height*scale)
  new_width = tf.to_int32(width*scale)
  image = tf.image.resize_images(image, (new_height, new_width), method=0)
  image = tf.image.resize_image_with_crop_or_pad(image, max_side, max_side)
  return image


def load_step1(g, pb_path, sess):
    with sess.as_default():
      with g.as_default():
        gx = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
          serialized_graph = fid.read()
          gx.ParseFromString(serialized_graph)
          tf.import_graph_def(gx, name='')
        step1_inputs = g.get_tensor_by_name('image_tensor:0')
        step1_scores = tf.squeeze(g.get_tensor_by_name('detection_scores:0'), [0])
        step1_boxes = tf.squeeze(g.get_tensor_by_name('detection_boxes:0'), [0])
        step1_classes = tf.squeeze(g.get_tensor_by_name('detection_classes:0'), [0])
        return step1_inputs,step1_boxes,step1_classes,step1_scores
    
def load_step2(g, pb_path, sess):
    with sess.as_default():
      with g.as_default():
        gx = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
          serialized_graph = fid.read()
          gx.ParseFromString(serialized_graph)
          tf.import_graph_def(gx, name='')
        step2_inputs = g.get_tensor_by_name('input_node:0')
        step2_scores = g.get_tensor_by_name('output:0')
        return step2_inputs,step2_scores
            
def step1_result_process(detect_dict, thr=0.8):
    h,w,_ = detect_dict['image_rgb'].shape
    area = np.where(detect_dict['step1_scores']>thr)
    detect_dict['step1_scores'] = detect_dict['step1_scores'][area]
    detect_dict['step1_classes'] = detect_dict['step1_classes'][area]
    detect_dict['step1_boxes'] = detect_dict['step1_boxes'][area]
    if len(detect_dict['step1_scores'])>0:
        box = detect_dict['step1_boxes']
        detect_dict['step1_boxes'] = box*[h,w,h,w]
    return detect_dict

def nms(box,scores, max_ovr=0.5):
    x1 = box[:,1]
    y1 = box[:,0]
    x2 = box[:,3]
    y2 = box[:,2]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter/(areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= max_ovr)[0]
        order = order[inds + 1]
    return keep

def step2_result_nms(detect_dict, max_ovr=0.5):
    keeps_mask = np.ones(len(detect_dict['step2_classes']))
    e = list(set(detect_dict['step2_classes']))
    num = len(e)
    if num < len(detect_dict['step1_classes']):
        for i in range(num):
            cls = e[i]
            area = np.where(detect_dict['step2_classes']==cls)
            if len(area[0])==1:
                continue
            box_i = detect_dict['step1_boxes'][area]
            score_i = detect_dict['step2_scores'][area]
            keep = nms(box_i, score_i, max_ovr)
            for j in range(len(area[0])):
                if j not in keep:
                    keeps_mask[area[0][j]]=0
    detect_dict['step2_nms_keeps']=keeps_mask
    return detect_dict

def draw_image(detect_dict):
    image = detect_dict['image_bgr']
    for i in range(len(detect_dict['step1_classes'])):
        if detect_dict['step2_nms_keeps'][i]==1:
            box = detect_dict['step1_boxes'][i]
            cv2.rectangle(image, (int(box[1]),int(box[0])), (int(box[3]),int(box[2])), (0,0,255) ,1)
            text1 = 'box: %f'%detect_dict['step1_scores'][i]
            text2 = '%s: %f'%(detect_dict['step2_skus'][i], detect_dict['step2_scores'][i])
            cv2.putText(image, text1, (int(box[1]),int(box[0])+10), 0, 0.5, (0,0,0),1)
            cv2.putText(image, text2, (int(box[1]),int(box[0])+30), 0, 0.5, (255,0,0),1)
    return detect_dict

