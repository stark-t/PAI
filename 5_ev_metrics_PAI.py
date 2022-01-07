"""
Created on 5th of Jan. 2022
@author: Robin, Thomas

README:

- change variables in lines 15-28
"""

import os
import detect
import shutil
import tqdm
import numpy as np
from utils.metrics import bbox_iou
import torch
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

#decide if you want bounding boxes in the detected images
img_bounding_boxes = True
#decide if you want to delete the predictions (all images from source with predictions) (for checkup for example)
delete_prediction_images = True

conf_threshold = .50
batch_size = 16
imgsz = 1280
datasetname = "UFZ_field_observation_21_08"
source = r"C:\MASTERTHESIS\Data\test_dataset_for_ev_metrics\test\images"
save_dir = r"C:\MASTERTHESIS\Results\insect_detector"
# save_dir = r"C:\MASTERTHESIS\Results\pollinator_detector"
# save_dir = r"C:\MASTERTHESIS\Results\order_classification"
weights = r"C:\MASTERTHESIS\Results\Training\Trial_insect_detector_200_yolov5m6_1280\weights\best.pt"

"""


"""

#make folder to save predictions if not exist
modelname = weights.split("\\")[4]
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
for file_name in tqdm.tqdm(os.listdir(labels_dir)):
    #get path to label and prediction file
    label_file = os.path.join(labels_dir, file_name)
    prediction_file = os.path.join(prediction_dir, file_name)
    #read and get label and prediction
    with open(label_file) as lf:
        label_info = lf.readlines()
        labels = []
        for i, info in enumerate(label_info):
            label_info_ = label_info[i].split(" ")
            labels.append(label_info_)
        labels = np.array(labels)
    with open(prediction_file) as pf:
        prediction_info = pf.readlines()
        predictions = []
        for i, info in enumerate(prediction_info):
            prediction_info_ = prediction_info[i].split(" ")
            predictions.append(prediction_info_)
        predictions = np.array(predictions)
    if len(labels) != len(predictions):
        problem = 1
    if len(labels) > 1:
        debug = 1
    ious = []
    for label_i, label in enumerate(labels):
        all_ious_per_label = []
        for prediction_i, prediction in enumerate(predictions):
            bbox_label = label.astype(np.float32)
            bbox_label_torch = torch.from_numpy(bbox_label)
            bbox_prediction = prediction.astype(np.float32)
            bbox_prediction_torch = torch.from_numpy(bbox_prediction)
            if len(labels) > 1:
                # p1 = Polygon([[label[1], label[2]], [label[3], label[2]], [label[3], label[4]], [label[1], label[4]]], closed=False, color="red", alpha=0.5)
                # p2 = Polygon([[prediction[1], prediction[2]], [prediction[3], prediction[2]], [prediction[3], prediction[4]], [prediction[1], prediction[4]]], closed=False, color="blue", alpha=0.5)
                # ax = plt.gca()
                # ax.add_patch(p1)
                # ax.add_patch(p2)
                # plt.show()
                debug = 1
            iou_torch = bbox_iou(bbox_label_torch[1:], bbox_prediction_torch[1:])
            iou = iou_torch.numpy()
            iou2 = get_iou(bbox_label[1:], bbox_prediction[1:])
            #vergleiche alle predictions pro label
            all_ious_per_label.append(iou)
        iou_max_per_label = np.max(all_ious_per_label)
        ious.append(iou_max_per_label)
    debug = 1
        # if len(labels) > 1:
        #     debug = 1

    # weitere evaluation metrics? später!

    #wie ist es mit mehreren predictions pro Bild?
    #was ist wenn anzahl predictions ungleich anzahl labels?







#delete redundant images in exp
if delete_prediction_images:
    for image_to_delete in tqdm.tqdm(os.listdir(os.path.join(save_dir, 'exp'))):
        if image_to_delete.endswith('.jpg'):
            delete_redundant_image = os.path.join(save_dir, 'exp', image_to_delete)
            delete_redundant_image = delete_redundant_image.replace('.txt', '.jpg')
            os.remove(delete_redundant_image)
        elif image_to_delete.endswith('.JPG'):
            delete_redundant_image = os.path.join(save_dir, 'exp', image_to_delete)
            delete_redundant_image = delete_redundant_image.replace('.txt', '.JPG')
            os.remove(delete_redundant_image)

#rename outputfile
newfoldername = os.path.join(save_dir, (datasetname+'_'+'Threshhold_'+str(int(conf_threshold*100))+'%'))
os.rename(os.path.join(save_dir, 'exp'), newfoldername)