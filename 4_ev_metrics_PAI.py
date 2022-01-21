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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, cohen_kappa_score
from utils.metrics import bbox_iou
import seaborn as sns
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

#decide if you want bounding boxes in the detected images #! TODO: Ist die Bedingungen hier schon drin? Brauchen wir das überhaupt?
img_bounding_boxes = True
#decide if you want to delete the predictions (all images from source with predictions) (for checkup for example)
delete_prediction_images = True

conf_threshold = .50
batch_size = 16
imgsz = 1280

source = r"C:\MASTERTHESIS\Data\UFZ_2021_07_07_dataset_annotated_insect_detector\test\images"
# source = r"C:\MASTERTHESIS\Data\ArTaxOr_dataset_annotated_insect_detector\test\images"
# source = r"C:\MASTERTHESIS\Data\test_dataset_for_ev_metrics\test\images"
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

del iou_, iou, iou_torch
iou_mean = np.mean(ious)
iou_std = np.std(ious)
print('[INFO]    Mean overall IOU for {} of {} bounding boxes:    {:5.4f}   with an standard deviation of {:5.4f}'.format(len(ious), len(y_true), iou_mean, iou_std))

accuracy = accuracy_score(y_true, y_pred)
print('[INFO]    Accuracy:    {:5.4f}'.format(accuracy))

error_rate = 1 - accuracy #!TODO: Nochmal überprüfen ob das wirklich die error rate ist
print('[INFO]    Error Rate:    {:5.4f}'.format(error_rate))

kappa = cohen_kappa_score(y_true, y_pred)
print('[INFO]    Kappa Coefficient:    {:5.4f}'.format(kappa))

precision = precision_score(y_true, y_pred, average='macro')
print('[INFO]    Precision (macro):    {:5.4f}'.format(precision))

recall = recall_score(y_true, y_pred, average='macro')
print('[INFO]    Recall (macro):    {:5.4f}'.format(recall))

f1 = f1_score(y_true, y_pred, average='binary', pos_label=0) #!TODO Hier mal mit Verena sprechen über micro, weighted, ...
print('[INFO]    F1-score (macro):    {:5.4f}'.format(f1))

#Seaborn Confusion Matrix Plot:
cf_matrix = confusion_matrix(y_true, y_pred)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
matrix_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
matrix_labels = np.asarray(matrix_labels).reshape(2,2)
ax = sns.heatmap(cf_matrix, annot=matrix_labels, fmt='', cmap='Blues')
ax.set_title('Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Ground Truth');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

# show and save plot to folder
save_metrics_info_path = os.path.join(save_dir, "exp")
plt.savefig(os.path.join(save_metrics_info_path, '_Confusion_Matrix.png'))
plt.show()

# save metrics to text file
nameoffile = "_Evaluation_Metrics.txt"
completeName = os.path.join(save_metrics_info_path, nameoffile)
with open(completeName, "w") as file1:
    file1.write('[Mean overall IOU] for {} of {} bounding boxes:    {:5.4f}   with an [standard deviation] of {:5.4f}'.format(len(ious), len(y_true), iou_mean, iou_std) + "\n")
    file1.write('[Accuracy]:    {:5.4f}'.format(accuracy) + "\n")
    file1.write('[Error Rate]:    {:5.4f}'.format(error_rate) + "\n")
    file1.write('[Kappa Coefficient]:    {:5.4f}'.format(kappa) + "\n")
    file1.write('[Precision (Macro)]:    {:5.4f}'.format(precision) + "\n")
    file1.write('[Recall (Macro)]:    {:5.4f}'.format(recall) + "\n")
    file1.write('[F1-score (Macro)]:    {:5.4f}'.format(f1) + "\n" + "\n")
    file1.write('(Macro = Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.)')



#!TODO: alle metrics berechnen

