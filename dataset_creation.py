import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
from ultralytics import YOLO

'''
This script is used to create the dataset:
    - Organize the images in the folders
    - Create labels for object detection:
        - .txt files with format '0 x_c y_c w h'
'''

# TRAINING DATA
with open('/content/speedplusv2/synthetic/train.json') as f:
    data = f.read()
    
train_js = json.loads(data)

move_image(train_js, '/content/speedplusv2/synthetic/images/', '/content/speedplusv2/synthetic/images/train/')
for i, _ in enumerate(train_js):
    image_path = '/content/speedplusv2/synthetic/images/train/' + train_js[i]["filename"]
    tango_pos =  train_js[i]["r_Vo2To_vbs_true"]
    tango_qaud = train_js[i]["q_vbs2tango_true"]

    x, y = project(Tango.M, tango_qaud, tango_pos)
    save_labels(x, y, ('/content/speedplusv2/synthetic/labels/train/' + train_js[i]["filename"]).replace('jpg', 'txt'))
    if train_js[i]["filename"] == "img000001.jpg":
    
       plot_projection(tango_qaud, tango_pos, image_path)
       
# VALIDATION DATA
with open('/content/speedplusv2/synthetic/validation.json') as f:
    data = f.read()

val_js = json.loads(data)
move_image(val_js, '/content/speedplusv2/synthetic/images/', '/content/speedplusv2/synthetic/images/val/')
for i, _ in enumerate(val_js):
    image_path = '/content/speedplusv2/synthetic/images/val/' + val_js[i]["filename"]
    tango_pos =  val_js[i]["r_Vo2To_vbs_true"]
    tango_qaud = val_js[i]["q_vbs2tango_true"]

    x, y = project(Tango.M, tango_qaud, tango_pos)
    save_labels(x, y, ('/content/speedplusv2/synthetic/labels/val/' + val_js[i]["filename"]).replace('jpg', 'txt'))
    if val_js[i]["filename"] == "img000113.jpg":
              
       plot_projection(tango_qaud, tango_pos, image_path)


# SUNLAMP DATA
with open('/content/speedplusv2/sunlamp/test.json') as f:
    data = f.read()

sun_js = json.loads(data)

for i, _ in enumerate(sun_js):
    image_path = '/content/speedplusv2/sunlamp/images/' + val_js[i]["filename"]
    tango_pos =  sun_js[i]["r_Vo2To_vbs_true"]
    tango_qaud = sun_js[i]["q_vbs2tango_true"]

    x, y = project(Tango.M, tango_qaud, tango_pos)
    save_labels(x, y, ('/content/speedplusv2/sunlamp/labels/' + sun_js[i]["filename"]).replace('jpg', 'txt'))

    
# LIGHTBOX DATA
with open('/content/speedplusv2/lightbox/test.json') as f:
    data = f.read()

light_js = json.loads(data)

for i, _ in enumerate(light_js):
    image_path = '/content/speedplusv2/lightbox/images/' + val_js[i]["filename"]
    tango_pos =  light_js[i]["r_Vo2To_vbs_true"]
    tango_qaud = light_js[i]["q_vbs2tango_true"]

    x, y = project(Tango.M, tango_qaud, tango_pos)
    save_labels(x, y, ('/content/speedplusv2/lightbox/labels/' + light_js[i]["filename"]).replace('jpg', 'txt'))