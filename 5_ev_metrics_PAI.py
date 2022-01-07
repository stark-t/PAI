"""
Created on 5th of Jan. 2022
@author: Robin, Thomas

README:

- change variables in lines 15-28
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import detect
import shutil
import tqdm
import numpy as np
from utils.metrics import bbox_iou
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

verbose = 0

#decide if you want bounding boxes in the detected images
img_bounding_boxes = True
#decide if you want to delete the predictions (all images from source with predictions) (for checkup for example)
delete_prediction_images = True

conf_threshold = .50
batch_size = 16
imgsz = 1280

source = r"D:\202105_PAI\data\test_dataset_for_ev_metrics\test\images"
save_dir = r"D:\202105_PAI\data\test_dataset_for_ev_metrics\results"
weights = r"D:\202105_PAI\data\best.pt"


#make folder to save predictions if not exist
modelname = weights.split("\\")[3]
sourcename = source.split("\\")[3]
save_dir = os.path.join(save_dir, (modelname + "_" + sourcename))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#delete exp folder if exists
if os.path.exists(os.path.join(save_dir, "exp")):
    shutil.rmtree(os.path.join(save_dir, "exp"))

#run yolo to detect images
output = detect.run(weights=weights, source=source, imgsz=(imgsz, imgsz), save_txt=True, conf_thres=conf_threshold, project=save_dir)

#rename folders
prediction_dir = os.path.join(save_dir, 'exp', 'predictions')
if not os.path.exists(prediction_dir):
    os.rename(os.path.join(save_dir, 'exp', 'labels'), prediction_dir)

#copy labels into exp folder
source_labels = source.replace("images", "labels")
labels_dir = os.path.join(save_dir, 'exp', "labels")
if not os.path.exists(labels_dir):
    shutil.copytree(source_labels, labels_dir, dirs_exist_ok=True)

#loop through all labels
list_of_all_ious = []
for file_number, file_name in tqdm.tqdm(enumerate(os.listdir(labels_dir))):

    #get path to label and prediction file
    label_file = os.path.join(labels_dir, file_name)
    prediction_file = os.path.join(prediction_dir, file_name)

    #read and get label and prediction
    with open(label_file) as lf:
        label_lines_str = lf.readlines()
        labels = []
        for i, info in enumerate(label_lines_str):
            label_info_str = label_lines_str[i].split(" ")
            label_floats = [float(f) for f in label_info_str]
            labels.append(label_floats)

    with open(prediction_file) as pf:
        prediction_lines_str = pf.readlines()
        predictions = []
        for i, info in enumerate(prediction_lines_str):
            prediction_info_str = prediction_lines_str[i].split(" ")
            prediction_floats = [float(f) for f in prediction_info_str]
            predictions.append(prediction_floats)

    ious = []
    for label_i, label in enumerate(labels):
        all_ious_per_label = []
        for prediction_i, prediction in enumerate(predictions):
            bbox_label = np.array(label[1:])
            bbox_label_torch = torch.from_numpy(bbox_label)

            bbox_prediction = np.array(prediction[1:])
            bbox_prediction_torch = torch.from_numpy(bbox_prediction)

            iou_torch = bbox_iou(bbox_label_torch, bbox_prediction_torch, x1y1x2y2=False)
            iou = iou_torch.numpy()

            if iou < 0:
                iou = 0.0
            elif iou > 1.0:
                if verbose == 1:
                    print('Warning: Ill defined IOU of above 1.0')
                iou = 1.0

            # for all predictions im images calculate iou for i-th label
            all_ious_per_label.append(iou)

        # get highest iou per label per prediction
        iou_max_per_label = np.max(all_ious_per_label)
        ious.append(iou_max_per_label)

    # get mean iou per images
    mean_iou_per_images = np.mean(ious)
    list_of_all_ious.append(mean_iou_per_images)

# get mean overall iou
mean_overall_iou = np.mean(list_of_all_ious)
print('[INFO]    Mean overall IOU:    {:5.4f}   with an standart deviation of {:5.4f}'.format(mean_overall_iou, np.std(list_of_all_ious)))


