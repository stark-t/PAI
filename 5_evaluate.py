# import statements
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, cohen_kappa_score, matthews_corrcoef, classification_report
from utils.metrics import bbox_iou
import seaborn as sns
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob

# base_dir = r'C:\MASTERTHESIS\Data\P1_orders\test'
# base_dir = r'C:\MASTERTHESIS\Data\Field_observation\test'
# base_dir = r'C:\MASTERTHESIS\Data\Holiday_images\test'
base_dir = r'C:\MASTERTHESIS\Data\..Challenges\new\try'

prediction_dir = os.path.join(base_dir, 'results', 'exp', 'predictions')
labels_dir = os.path.join(base_dir, 'labels')
images_dir = os.path.join(base_dir, 'images')

labels_dirs = glob.glob(labels_dir + '\*')
images_dirs = glob.glob(images_dir + '\*')

print('\nThere are {} images and {} labeled images'.format(len(images_dirs), len(labels_dirs)))

y_true = []
y_pred = []
ious = []
ious_correct = []
classes = []
n_BB = 1

#loop through all labels
print('... Calculate metrics for each image')
for image_count_index, image_path in enumerate(images_dirs):

    ious_ = []
    iou_max_indicies = []

    # get path to label and prediction file
    file_name = image_path.split('\\')[-1]
    file_name = file_name.split('.')[0]
    label_file = os.path.join(labels_dir, (file_name + '.txt'))
    prediction_file = os.path.join(prediction_dir, (file_name + '.txt'))

    # if label file exists read bb and class
    labels = []
    if os.path.exists(label_file):
        with open(label_file) as lf:
            label_lines_str = lf.readlines()
            for i, info in enumerate(label_lines_str):
                label_info_str = label_lines_str[i].split(" ")
                label_floats = [float(f) for f in label_info_str]
                labels.append(label_floats)
                n_BB += 1

                class_ = label_floats[0]
                classes.append(class_)
    else:
        # Case -1: there is no label for the imagefile
        labels.append([-1,0,0,0,0])

    # if prediction file exists read bb and class
    predictions = []
    if os.path.exists(prediction_file):
        with open(prediction_file) as pf:
            prediction_lines_str = pf.readlines()
            for i, info in enumerate(prediction_lines_str):
                prediction_info_str = prediction_lines_str[i].split(" ")
                prediction_floats = [float(f) for f in prediction_info_str]
                predictions.append(prediction_floats)
    else:
        # Case -2: there is no prediction for the imagefile
        predictions.append([-2,0,0,0,0])

    # get predictions and labels into the same lengths using dummy values
    if len(labels) > len(predictions):
        diff_len = len(labels) - len(predictions)
        for i in range(diff_len):
            # Case -3: there are more labels than prediction in the imagefile
            predictions.append([-3,0,0,0,0])
    elif len(labels) < len(predictions):
        diff_len = len(predictions) - len(labels)
        for i in range(diff_len):
            # Case -4: there are more predictions than labels in the imagefile
            labels.append([-4,0,0,0,0])

    # Go through prediction and label to calculate all possible iou boundingbox matches
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
        ious.append(iou)

        # get index of
        iou_max_i = iou_.index(max(iou_))
        iou_max_indicies.append(iou_max_i)

    while 0 in ious:
        ious.remove(0)

    if label[0] == -1:
        debug = 1

    # get matching classes
    for i, class_i in enumerate(iou_max_indicies):
        prediction = predictions[i]
        y_pred_ = int(prediction[0])
        y_pred.append(y_pred_)
        if y_pred_ >= 0:
            label = labels[class_i]
            y_true_ = int(label[0])
            y_true.append(y_true_)
            # ious_correct.append(ious[i]) #! TODO: Die Anzahl der richtigen bounding boxes muss noch errechnet werden!!
        else:
            label = labels[i]
            y_true_ = int(label[0])
            y_true.append(y_true_)

print('In {} labels are {} objects with BBs'.format(len(labels_dirs), n_BB))
unique_classes, unique_counts = np.unique(classes, return_counts=True)
for class_ in unique_classes:
    print('For class {} there are {} BBs'.format(int(class_), unique_counts[int(class_)]))
print('From {} labels {} were correctly identified'.format(n_BB, len(ious)))

el_y_true = np.unique(y_true, return_counts=True)
el_y_pred = np.unique(y_pred, return_counts=True)

y_true = [0 if f < 0 else f+1 for f in y_true]
y_pred = [0 if f < 0 else f+1 for f in y_pred]

del iou_, iou, iou_torch
iou_mean = np.mean(ious)
iou_std = np.std(ious)
print('[INFO]    Mean overall IOU for {} of {} bounding boxes: {:5.4f} with an standard deviation of {:5.4f}'.format(len(ious), n_BB, iou_mean, iou_std))

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

f1 = f1_score(y_true, y_pred, average='weighted')
print('[INFO]    F1-score (weighted):    {:5.4f}'.format(f1))


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

    # fix confusion matrix for plot
    cm = confusion_matrix(y_true, y_pred)
    # cm[0] = 0
    # cm[0, 0] = 1
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
    # plt.savefig(os.path.join(r'C:\MASTERTHESIS\Data\P1_orders\test\results\exp\0_Confusion_Matrix.png'), dpi=400)
    plt.show()

#New Confusion Matrix (from function above):
classnames = ['Araneae', 'Coleoptera', 'Diptera', 'Hemiptera', 'Hymenoptera', 'Hymenoptera f.', 'Lepidoptera', 'Orthoptera']
# classnames = ['Araneae', 'Diptera', 'Hemiptera', 'Hymenoptera', 'Hymenoptera f.', 'Lepidoptera', 'Orthoptera']
# classnames = ['Insect']
cm_analysis(y_true, y_pred, ['background'] + classnames, ymap=None, figsize=(12,9))


