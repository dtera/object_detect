import os

import matplotlib.image as mpimg
import pandas as pd
from flask import Flask, request, render_template

from tf import tf_load_model_and_predict as lmp, visualization as vs

app = Flask(__name__, static_url_path="")
root_path = os.getcwd()
model_dir = os.path.join(root_path, os.path.join("data", "model"))
# sess, ops = lmp.load_ssd_mobilenet_model(model_dir)
sess1, ops1 = lmp.load_phone_inference(model_dir)
sess2, ops2 = lmp.load_screen_inference(model_dir)

UPLOAD_FOLDER = os.path.join(root_path, "static/upload/img")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'ico', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(file_name):
    return '.' in file_name and \
           file_name.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def get_phone_detect_result(img):
    detection_labels, detection_classes, detection_scores, detection_boxes = \
        lmp.predict_by_phone_inference(sess1, ops1, img)
    if len(detection_scores) > 0:
        d_labels, d_classes, d_scores, d_boxes = lmp.predict_by_screen_inference(sess2, ops2, img)
        if len(d_scores) == 0:
            detection_labels = ["非碎屏" + l for l in detection_labels]
        else:
            detection_labels, detection_classes, detection_scores, detection_boxes = \
                d_labels, d_classes, d_scores, d_boxes
    return detection_boxes, detection_classes, detection_labels, detection_scores


@app.route("/")
def index():
    return render_template('index.html')


# @app.route("/od", methods=["GET", "POST"])
# def od():
#     if request.method == "POST":
#         img_file = request.files["img"]
#         if img_file and allowed_file(img_file.filename):
#             upload_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
#             img_file.save(upload_path)
#             img = mpimg.imread(upload_path)
#             detection_labels, detection_classes, detection_scores, detection_boxes = \
#                 lmp.predict_by_ssd_mobilenet(sess, ops, img)
#             save_path = str(upload_path.rsplit('.', 1)[0]) + "_boxed." + str(upload_path.rsplit('.', 1)[1])
#             print("save_path: {}".format(save_path))
#             vs.plt_bboxes(img, detection_labels, detection_classes, detection_scores, detection_boxes, save_path)
#             return render_template('od_result.html', file_name=os.path.basename(save_path))
#     return ""


@app.route("/phoneDetectResult", methods=["GET", "POST"])
def phone_detect_result():
    if request.method == "POST":
        img_file = request.files["img"]
        if img_file and allowed_file(img_file.filename):
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
            img_file.save(upload_path)
            img = mpimg.imread(upload_path)
            detection_boxes, detection_classes, detection_labels, detection_scores = get_phone_detect_result(img)
            # detection_boxes = list(map(lambda b: b.tolist(), detection_scores))
            data = {"detection_boxes": detection_boxes, "detection_classes": detection_classes,
                    "detection_labels": detection_labels, "detection_scores": detection_scores}
            return pd.Series(data).to_json(force_ascii=False)
    return ""


@app.route("/phoneDetect", methods=["GET", "POST"])
def phone_detect():
    if request.method == "POST":
        img_file = request.files["img"]
        if img_file and allowed_file(img_file.filename):
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
            img_file.save(upload_path)
            img = mpimg.imread(upload_path)
            save_path = str(upload_path.rsplit('.', 1)[0]) + "_boxed." + str(upload_path.rsplit('.', 1)[1])
            print("save_path: {}".format(save_path))
            detection_boxes, detection_classes, detection_labels, detection_scores = get_phone_detect_result(img)
            vs.plt_bboxes(img, detection_labels, detection_classes, detection_scores, detection_boxes, save_path)
            return render_template('od_result.html', file_name=os.path.basename(save_path))
    return ""


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
