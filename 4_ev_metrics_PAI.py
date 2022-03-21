"""
Created on 5th of Jan. 2022
@author: Robin, Thomas

README:

- make adjustments to lines 10-22
"""

#decide if you want bounding boxes in the detected images #! TODO: Ist die Bedingungen hier schon drin? Braucht Valentin das?
img_bounding_boxes = True
#decide if you want to delete the predictions (all images from source with predictions) (for checkup for example)
delete_prediction_images = False

conf_threshold = .50
batch_size = 16
imgsz = 1280
classnames = ['Araneae','Diptera', 'Hemiptera', 'Hymenoptera f.', 'Hymenoptera', 'Lepidoptera', 'Orthoptera']
# classnames = ['Insect']
# source = r"C:\MASTERTHESIS\Data\P1_beta_orders\test\images"
source = r"C:\MASTERTHESIS\Data\Testdatensatz_Programming\test\images"
save_dir = r"C:\MASTERTHESIS\Results\Evaluation"
weights = r"C:\MASTERTHESIS\Results\Training\P1_beta_orders_200_yolov5m6\weights\best.pt"

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import detect
import shutil
import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, cohen_kappa_score, matthews_corrcoef, classification_report
from utils.metrics import bbox_iou
import seaborn as sns
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Confusion Matrix Function:
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(12,9)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred) #, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    # cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = pd.DataFrame(cm_perc, index=labels, columns=labels)
    # cm.index.name = 'Actual'
    # cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='Blues', square=True)
    ax.set_xlabel('\nPredicted');
    ax.set_ylabel('Actual\n');
    ax.figure.tight_layout()
    ax.figure.subplots_adjust(bottom=0.2)
    #plt.savefig(filename)
    plt.savefig(os.path.join(save_metrics_info_path, '_Confusion_Matrix.png'), dpi=250)
    plt.show()

#make folder to save predictions if not exist
sourcename = source.split("\\")[3]
weightsname = weights.split("\\")[4]
save_dir = os.path.join(save_dir, (sourcename + "_" + weightsname + "_best_" + 'Threshhold_' + str(int(conf_threshold*100)) + '%'))
save_metrics_info_path = os.path.join(save_dir, "exp")
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

    # get predictions and labels into the same lengths using dummy values
    if len(labels) > len(predictions):
        diff_len = len(labels) - len(predictions)
        for i in range(diff_len):
            predictions.append([-1,0,0,0,0])
    elif len(labels) < len(predictions):
        diff_len = len(predictions) - len(labels)
        for i in range(diff_len):
            labels.append([-1,0,0,0,0])

    # Go through prediction and label to get fitting metrics
    for label_i, label in enumerate(labels):
        iou_ = []
        for prediction_i, prediction in enumerate(predictions):

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

y_pred.append(-1)
y_true.append(-1)

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

kappa = cohen_kappa_score(y_true, y_pred)
print('[INFO]    Kappa Coefficient:    {:5.4f}'.format(kappa))

matthews = matthews_corrcoef(y_true, y_pred)
print('[INFO]    Matthews Correlation Coefficient:    {:5.4f}'.format(matthews))

precision = precision_score(y_true, y_pred, average='weighted')
print('[INFO]    Precision (weighted):    {:5.4f}'.format(precision))

recall = recall_score(y_true, y_pred, average='weighted')
print('[INFO]    Recall (weighted):    {:5.4f}'.format(recall))

f1 = f1_score(y_true, y_pred, average='weighted') #!TODO Hier mal mit Verena sprechen über micro, weighted, ...
print('[INFO]    F1-score (weighted):    {:5.4f}'.format(f1))

print('[INFO]    Classification Report:' + "\n")
print(classification_report(y_true, y_pred, target_names=['background'] + classnames))

#New Confusion Matrix (from function above):
cm_analysis(y_true, y_pred, ['background'] + classnames, ymap=None, figsize=(12,9))

#Seaborn Confusion Matrix Plot:
# cf_matrix = confusion_matrix(y_true, y_pred)
# group_percentages = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
# plt.subplots(figsize=(12,9))
# ax = sns.heatmap(group_percentages, annot=True, fmt='.2f', cmap='Blues', square=True)
# ax = sns.heatmap(cf_matrix, annot=labels_matrix, fmt='', cmap='Blues', square=True)
# ax = sns.heatmap(group_percentages, annot=True, fmt='.2f', cmap='viridis', square=True)
# ax.set_title('Confusion Matrix\n');
# ax.set_xlabel('\nPredicted');
# ax.set_ylabel('Actual\n');
# ax.figure.tight_layout()
# ax.figure.subplots_adjust(bottom = 0.2)

## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['background FN'] + classnames, rotation = 90)
# ax.yaxis.set_ticklabels(['background FP'] + classnames, rotation = 0)

# show and save plot to folder
# save_metrics_info_path = os.path.join(save_dir, "exp")
# plt.savefig(os.path.join(save_metrics_info_path, '_Confusion_Matrix.png'), dpi=250)
# plt.show()

# save metrics to text file
nameoffile = "_Evaluation_Metrics.txt"
completeName = os.path.join(save_metrics_info_path, nameoffile)
with open(completeName, "w") as file1:
    file1.write('[Mean overall IOU] for {} of {} bounding boxes:    {:5.4f}   with an [standard deviation] of {:5.4f}'.format(len(ious), len(y_true), iou_mean, iou_std) + "\n")
    file1.write('[Accuracy]:    {:5.4f}'.format(accuracy) + "\n")
    file1.write('[Kappa Coefficient]:    {:5.4f}'.format(kappa) + "\n")
    file1.write('[Matthews Correlation Coefficient]:    {:5.4f}'.format(matthews) + "\n")
    file1.write('[Precision (Weighted)]:    {:5.4f}'.format(precision) + "\n")
    file1.write('[Recall (Weighted)]:    {:5.4f}'.format(recall) + "\n")
    file1.write('[F1-score (Weighted)]:    {:5.4f}'.format(f1) + "\n")
    file1.write('[Classification Report]:' + "\n" + "\n")
    file1.write(classification_report(y_true, y_pred, target_names= ['background'] + classnames))

#!TODO: alle metrics berechnen

