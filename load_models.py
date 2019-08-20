#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import math

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

get_ipython().run_line_magic('matplotlib', 'inline')



# load model path
MODEL_NAME = './ssd_mobile_recognize_broken_screen_v1/pbFile'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('./models', 'object_detection.pbtxt')


# ### 加载计算图

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ###  参数设置以及标签转化
NUM_CLASSES = 1
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ### 图片加载
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# ### 限定图片大小
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# ###  加载图片到模型
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session(config=config) as sess:
      K.set_session(sess)
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
      #sess.close()
  return output_dict


# ### 可视化以及计算IOU的代码
image_ratio=[]
#PATH_TO_TEST_IMAGES_DIR="./test_datasets_second/datasets/newImage/"
PATH_TO_TEST_IMAGES_DIR="./dataPhone_v2/crop/"

for filetype in ['broken','nonbroken']:
    for file in glob(PATH_TO_TEST_IMAGES_DIR+filetype+'/*.png'):
    #for index in range(len(filename)):
      #image = Image.open(PATH_TO_TEST_IMAGES_DIR+filename[index].decode('utf-8'))
      image = Image.open(file)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)
          #min_score_thresh=0.6)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      plt.axis('off') # 不显示坐标轴
    #   plt.savefig("./test_datasets_second/picture_result/"+filename[index].decode('utf-8'))
      plt.savefig(file.replace("crop/"+filetype,"crop/result"))

"""
    extract the roi:提取图片区域
"""
#   final_score=np.squeeze(output_dict['detection_scores'])
#   count=0
#   for i in range(90):
#     if output_dict['detection_scores'] is None or final_score[i] > 0.5:
#         count+=1

#   im_width,im_height=image.size
    
# #   y_min = 0
# #   x_min = 0 
# #   y_max = im_height
# #   x_max = im_width 
    
#   #for i in range(count):
#   y_min = math.floor(output_dict['detection_boxes'][0][0]*im_height)
#   x_min = math.floor(output_dict['detection_boxes'][0][1]*im_width)
#   y_max = math.ceil(output_dict['detection_boxes'][0][2]*im_height)
#   x_max = math.ceil(output_dict['detection_boxes'][0][3]*im_width)

#   box2=[x_min,y_min,x_max,y_max]

#   ori_y_min = math.floor(ymin[index]*im_height)
#   ori_x_min = math.floor(xmin[index]*im_width)
#   ori_y_max = math.ceil(ymax[index]*im_height)
#   ori_x_max = math.ceil(xmax[index]*im_width)
#   box1=[ori_x_min,ori_y_min,ori_x_max,ori_y_max]

#   ratio=calc_iou(box1, box2)
#   image_ratio.append(ratio)
#   #cv2.imencode('.jpg', roi)[1].tofile("./picture_result/bad/"+image_path)