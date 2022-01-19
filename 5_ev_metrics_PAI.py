"""
Created on 5th of Jan. 2022
@author: Robin, Thomas

README:

- change variables in lines
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import detect
import shutil
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from utils.metrics import bbox_iou
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

#decide if you want bounding boxes in the detected images
img_bounding_boxes = True
#decide if you want to delete the predictions (all images from source with predictions) (for checkup for example)
delete_prediction_images = True

conf_threshold = .50
batch_size = 16
imgsz = 1280

source = r"C:\MASTERTHESIS\Data\ArTaxOr_dataset_annotated_insect_detector\test\images"
save_dir = r"C:\MASTERTHESIS\Results\Evaluation"
weights = r"C:\MASTERTHESIS\Results\Training\Trial_insect_detector_200_yolov5m6\weights\best.pt"


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


y_true = []
y_pred = []
ious = []

#loop through all labels
print('[INFO]:    calculate metrics for each image')
for file_number, file_name in tqdm.tqdm(enumerate(os.listdir(labels_dir))):

    ious_ = []
    iou_max_indicies = []

    #get path to label and prediction file
    label_file = os.path.join(labels_dir, file_name)
    prediction_file = os.path.join(prediction_dir, file_name)

    #read and get label and prediction
    labels = []
    if os.path.exists(label_file):
        with open(label_file) as lf:
            label_lines_str = lf.readlines()
            for i, info in enumerate(label_lines_str):
                label_info_str = label_lines_str[i].split(" ")
                label_floats = [float(f) for f in label_info_str]
                labels.append(label_floats)
    else:
        labels.append([-1,0,0,0,0])
    predictions = []
    if os.path.exists(prediction_file):
        with open(prediction_file) as pf:
            prediction_lines_str = pf.readlines()
            for i, info in enumerate(prediction_lines_str):
                prediction_info_str = prediction_lines_str[i].split(" ")
                prediction_floats = [float(f) for f in prediction_info_str]
                predictions.append(prediction_floats)
    else:
        predictions.append([-1,0,0,0,0])

    # get predictions and labels into the same lenght using dummy values
    if len(labels) > len(predictions):
        diff_len = len(labels) - len(predictions)
        for i in range(diff_len):
            predictions.append([-1,0,0,0,0])
    elif len(labels) < len(predictions):
        diff_len = len(predictions) - len(labels)
        for i in range(diff_len):
            labels.append([-1,0,0,0,0])

    # Go through prediction and label to get fitting metrics
    for prediction_i, prediction in enumerate(predictions):

        iou_ = []
        for label_i, label in enumerate(labels):

            bbox_label = np.array(label[1:])
            bbox_label_torch = torch.from_numpy(bbox_label)

            bbox_prediction = np.array(prediction[1:])
            bbox_prediction_torch = torch.from_numpy(bbox_prediction)

            iou_torch = bbox_iou(bbox_label_torch, bbox_prediction_torch, x1y1x2y2=False)
            iou = iou_torch.numpy()

            # for all predictions im images calculate iou for i-th label
            iou_.append(iou)

        # get highest iou per label per prediction
        iou = np.max(iou_)
        # ious_.append(iou)
        ious.append(iou)

        # get index
        iou_max_i = iou_.index(max(iou_))
        iou_max_indicies.append(iou_max_i)

    while 0 in ious:
        ious.remove(0)


    # get matching classes
    for i, class_i in enumerate(iou_max_indicies):
        label = labels[class_i]
        y_true_ = int(label[0])
        y_true.append(y_true_)

        prediction = predictions[i]
        y_pred_ = int(prediction[0])
        y_pred.append(y_pred_)



del iou_, iou, iou_torch
iou_mean = np.mean(ious)
iou_std = np.std(ious)
print('[INFO]    Mean overall IOU for {} of {} bounding boxes:    {:5.4f}   with an standard deviation of {:5.4f}'.format(len(ious), len(y_true), iou_mean, iou_std))

accuracy = accuracy_score(y_true, y_pred)
print('[INFO]    Accuracy:    {:5.4f}'.format(accuracy))

f1 = f1_score(y_true, y_pred, average='macro') #!TODO Hier mal mit Verena sprechen über micro, weighted, ...
print('[INFO]    F1-score:    {:5.4f}'.format(f1))

# save metrics to text file
save_metrics_path = os.path.join(save_dir, "exp")
nameoffile = "Evaluation Metrics.txt"
completeName = os.path.join(save_metrics_path, nameoffile)
with open(completeName, "w") as file1:
    file1.write('[Mean overall IOU] for {} of {} bounding boxes:    {:5.4f}   with an [standard deviation] of {:5.4f}'.format(len(ious), len(y_true), iou_mean, iou_std) + "\n")
    file1.write('[Accuracy]:    {:5.4f}'.format(accuracy) + "\n")
    file1.write('[F1-score]:    {:5.4f}'.format(f1) + "\n")



#!TODO: alle metrics berechnen
#!TODO: Funktion einfügen wenn classlabel ungleich predictionlabel ist soll er die iou NICHT berechnen für dieses objekt/insekt

