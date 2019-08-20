#!/usr/bin/python
# -*- coding:utf-8 -*-

import os

# from tensorflow.python.platform import gfile
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

from tf import visualization as vs


def load_model(model_dir, model_name, output_names, labels_file,
               scope="", model_suffix=".pb", var_suffix=":0"):
    sess = tf.compat.v1.Session()
    with tf.gfile.GFile(os.path.join(model_dir, model_name + model_suffix), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name=model_name)
    sess.run(tf.compat.v1.global_variables_initializer())

    for tensor in tf.contrib.graph_editor.get_tensors(tf.compat.v1.get_default_graph()):
        print(tensor, tensor.shape)
    # name_scope = sess.graph.get_name_scope()
    # collection_keys = sess.graph.get_all_collection_keys()
    # operations = sess.graph.get_operations()
    # print("name_scope: {}".format(name_scope))
    # print("collection_keys: {}".format(collection_keys))
    # print("operations: {}".format(operations))

    ops = []
    scope = model_name if not scope else scope
    for output_name in output_names:
        name = ((scope + "/" + output_name) if scope else output_name) + var_suffix
        output_tensor = sess.graph.get_tensor_by_name(name)
        ops.append(output_tensor)
    # print("ops: {}".format(ops))

    with open(os.path.join(model_dir, labels_file), "r") as f:
        labels = f.readlines()
        ops.append(list(map(lambda l: l.rstrip('\n'), labels)))
        # print(labels, len(labels))
    ops.append(model_dir)
    ops.append(scope)
    ops.append(var_suffix)
    return sess, ops


def predict(sess, ops, input_dict):
    feed_dict = {}
    scope = ops[-2]
    var_suffix = ops[-1]
    for input_name in input_dict:
        name = ((scope + "/" + input_name) if scope else input_name) + var_suffix
        input_tensor = sess.graph.get_tensor_by_name(name)
        feed_dict[input_tensor] = input_dict[input_name]
    # print("feed_dict: {}".format(feed_dict))
    # res = sess.run(output_names, feed_dict=input_dict)
    res = sess.run(ops[:-4], feed_dict=feed_dict)
    return res


def load_inception_model(model_dir):
    model_name = "tensorflow_inception_graph"
    output_names = ["output"]
    labels_file = "imagenet_comp_graph_label_strings.txt"
    return load_model(model_dir, model_name, output_names, labels_file)


def predict_by_inception(sess, ops, img):
    input_dict = {"input": np.reshape(img, [1] + list(img.shape))}
    return predict(sess, ops, input_dict)


def load_ssd_mobilenet_model(model_dir):
    model_name = "ssd_mobilenet_v1_android_export"
    output_names = ["detection_boxes", "detection_classes", "detection_scores", "num_detections"]
    labels_file = "coco_labels_list.txt"
    return load_model(model_dir, model_name, output_names, labels_file)


def predict_by_ssd_mobilenet(sess, ops, img):
    input_dict = {"image_tensor": np.reshape(img, [1] + list(img.shape))}
    labels = ops[-4]

    res = predict(sess, ops, input_dict)
    print("===================================output===================================")
    # [print(r, r.shape) for r in res]
    num_detections = int(res[3])
    detection_boxes = res[0][0][:num_detections]
    detection_classes = res[1][0][:num_detections]
    detection_labels = np.array(labels)[list(map(lambda c: int(c), detection_classes))]
    detection_scores = res[2][0][:num_detections]
    print("detection_boxes: {}".format(detection_boxes))
    print("detection_classes: {}".format(detection_classes))
    print("detection_labels: {}".format(detection_labels))
    print("detection_scores: {}".format(detection_scores))
    print("num_detections: {}".format(num_detections))
    return detection_labels, detection_classes, detection_scores, detection_boxes


def load_phone_inference(model_dir):
    model_name = "phone_inference_graph"
    output_names = ["detection_boxes", "detection_classes", "detection_scores", "num_detections"]
    labels_file = "phone_labels_list.txt"
    return load_model(model_dir, model_name, output_names, labels_file)


def predict_by_phone_inference(sess, ops, img):
    input_dict = {"image_tensor": np.reshape(img, [1] + list(img.shape))}
    labels = ops[-4]

    res = predict(sess, ops, input_dict)
    print("===================================output===================================")
    # [print(r, r.shape) for r in res]
    num_detections = int(res[3])
    detection_boxes = res[0][0][:num_detections]
    detection_classes = res[1][0][:num_detections]
    detection_labels = np.array(labels)[list(map(lambda c: int(c), detection_classes))]
    detection_scores = res[2][0][:num_detections]
    print("detection_boxes: {}".format(detection_boxes))
    print("detection_classes: {}".format(detection_classes))
    print("detection_labels: {}".format(detection_labels))
    print("detection_scores: {}".format(detection_scores))
    print("num_detections: {}".format(num_detections))
    return detection_labels, detection_classes, detection_scores, detection_boxes


def main():
    cur_path = os.getcwd()
    root_path = os.path.dirname(cur_path)
    model_dir = os.path.join(root_path, os.path.join("data", "model"))
    img_dir = os.path.join(root_path, os.path.join("data", "image"))
    save_dir = os.path.join(root_path, os.path.join(os.path.join("static", "upload"), "img"))
    image_names = sorted(os.listdir(img_dir))
    # print(image_names)
    image_name = image_names[-1]
    img = mpimg.imread(os.path.join(img_dir, image_name))

    # sess, ops = load_inception_model(model_dir)
    # res = predict_by_inception(sess, ops, img)
    # labels = ops[-4]
    # print("===================================output===================================")
    # max_idx = np.argmax(res[0][0])
    # print(res, res[0].shape, max_idx)
    # print(labels[max_idx], res[0][0][max_idx] * 100)

    sess, ops = load_ssd_mobilenet_model(model_dir)
    detection_labels, detection_classes, detection_scores, detection_boxes = \
        predict_by_ssd_mobilenet(sess, ops, img)

    # sess, ops = load_phone_inference(model_dir)
    # detection_labels, detection_classes, detection_scores, detection_boxes = \
    #     predict_by_phone_inference(sess, ops, img)

    save_path = os.path.join(save_dir, image_name.rsplit('.', 1)[0] + "_boxed." + image_name.rsplit('.', 1)[1])
    vs.plt_bboxes(img, detection_labels, detection_classes, detection_scores, detection_boxes, save_path)


if __name__ == "__main__":
    main()
