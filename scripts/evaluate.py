# import packages
import os
import glob
import shutil

import pandas as pd
import numpy as np
import torch
import tqdm as t
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from sklearn.metrics import accuracy_score

# import scripts
from detectors.yolov5.utils.metrics import bbox_iou
from scripts.utils_confusionmatrix import cm_analysis

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def calculate_ious(series):
    ious = []
    records = []
    for i in range(len(series['bbox_label'])):
        for j in range(len(series['bbox_pred'])):
            bbox_label = np.array(series['bbox_label'][i]).squeeze()
            bbox_label_torch = torch.from_numpy(bbox_label)
            bbox_label_torch = bbox_label_torch[None, :]

            bbox_prediction = np.array(series['bbox_pred'][j]).squeeze()
            bbox_prediction_torch = torch.from_numpy(bbox_prediction)
            bbox_prediction_torch = bbox_prediction_torch[None, :]

            if series['class_label'][i] < 0 or series['class_pred'][j] < 0:
                iou = -1.0
            else:
                iou_torch = bbox_iou(bbox_label_torch, bbox_prediction_torch)
                iou = iou_torch.numpy().squeeze()

            ious.append(iou)

            record = {
                'ID': series.ID,
                'file_pred': series.file_pred,
                'class_pred': series.class_pred[j],
                'bbox_pred': series.bbox_pred[j],
                'file_label': series.file_label,
                'class_label': series.class_label[i],
                'bbox_label': series.bbox_label[i],
                'iou': iou,
            }
            records.append(record)
    df = pd.DataFrame(records)
    # sort indicies based on iou value
    indicies_iou_sort_arr = np.argsort(ious)
    indicies_iou_sort_list = list(indicies_iou_sort_arr)
    indicies_iou_sort_list = list(reversed(indicies_iou_sort_list))
    #get highest iou values based on sort of comibantions
    indicies_iou = indicies_iou_sort_list[0:int(np.sqrt(len(df)))]
    df_iou_pairs = df.iloc[indicies_iou, :]

    return df_iou_pairs

def get_file_info(path_list):
    records = []
    for file in path_list:
        class_ids = []
        bboxes = []
        file_ids = []
        file_id = file.split(os.sep)[-1]
        file_id = file_id.split('.txt')[0]
        with open(file) as lf:
            label_lines_str = lf.readlines()
            for i, info in enumerate(label_lines_str):
                file_ids.append(file_id)
                label_info_str = label_lines_str[i].split(" ")[:5]
                label_floats = [float(f) for f in label_info_str]
                class_id = label_floats[0]
                class_ids.append(class_id)
                bbox = label_floats[1:]
                bboxes.append(bbox)
            record = {
                'ID': file_ids[0],
                'file': file,
                'class': class_ids,
                'bbox': bboxes,
            }
            records.append(record)
    return records

def run_evaluate(data_path='1', plot_cm=False):
    """

    :param input:
    :return:
    """

    predictions_list = glob.glob(data_path + os.sep + '*')
    if len(predictions_list) > 0:
        records = get_file_info(predictions_list)
        df_predictions = pd.DataFrame(records)
    else:
        df_predictions = pd.DataFrame(columns=['ID', 'file', 'class', 'bbox'])

    labels_list = glob.glob(r'F:\202105_PAI\data\P1_Data_sampled\test_valentin\labels' + os.sep + '*')
    records = get_file_info(labels_list)
    df_labels = pd.DataFrame(records)

    df = pd.merge(df_predictions, df_labels, on='ID', how="outer", suffixes=('_pred', '_label'))

    # replace nan in dataframe
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        if not isinstance(row['class_pred'], list):
            df.at[index, 'class_pred'] = [-1.0]
            df.at[index, 'bbox_pred'] = [[.0, .0, .0, .0]]
        if not isinstance(row['class_label'], list):
            df.at[index, 'class_label'] = [-1.0]
            df.at[index, 'bbox_label'] = [[.0, .0, .0, .0]]

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        # check if there are less labels than predictions
        if len(row['class_label']) < len(row['class_pred']):
            diff_len = np.abs(len(row['class_label']) - len(row['class_pred']))
            for i in range(diff_len):
                class_label = row['class_label']
                class_label.append(-1.0)
                df.at[index, 'class_label'] = class_label

                class_bbox = row['bbox_label']
                class_bbox.append([.0, .0, .0, .0])
                df.at[index, 'bbox_label'] = class_bbox

        # check if there are more labels than predictions
        elif len(row['class_label']) > len(row['class_pred']):
            diff_len = np.abs(len(row['class_label']) - len(row['class_pred']))
            for i in range(diff_len):
                class_pred = row['class_pred']
                class_pred.append(-1.0)
                df.at[index, 'class_pred'] = class_pred

                class_bbox = row['bbox_pred']
                class_bbox.append([.0, .0, .0, .0])
                df.at[index, 'bbox_pred'] = class_bbox

    # calclulate iou pairs
    records = []
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        df_iou_combination = calculate_ious(row)
        for _, record in df_iou_combination.iterrows():
            records.append(record.to_dict())

    df_ious = pd.DataFrame(records)
    df_ious_groupPercentile = df_ious.groupby('class_label')['iou'].agg([percentile(1), percentile(25), percentile(50),
                                                                         percentile(75), percentile(99)])
    print(df_ious_groupPercentile)


    y_true_list = list(df_ious['class_label'])
    y_true_list = [int(f) + 1 if f >= 0 else 0 for f in y_true_list]
    y_pred_list = list(df_ious['class_pred'])
    y_pred_list = [int(f) + 1 if f >= 0 else 0 for f in y_pred_list]

    labels = ['Background', 'Araneae', 'Coleoptera', 'Diptera', 'Hemiptera',
              'Hymenoptera_Formicidae', 'Hymenoptera', 'Lepidoptera', 'Orthoptera']

    results_path = data_path.split(os.sep)
    results_path = os.path.join(*results_path[:-1])
    filename = os.path.join(results_path, 'confusion_matrix.png')
    filename_tex = os.path.join(results_path, 'confusion_matrix_tex.txt')
    df_ious_groupPercentile.to_csv(os.path.join(results_path, 'class_ious_percentiles.csv'))
    if plot_cm:
        cm = cm_analysis(y_true_list, y_pred_list, labels, ymap=None, figsize=(9, 9),
                         filename=filename, filename_tex=filename_tex, plot='plot')
    else:
        cm = cm_analysis(y_true_list, y_pred_list, labels, ymap=None, figsize=(9, 9),
                         filename=filename, filename_tex=filename_tex, plot=None)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Specificity or true negative rate
    TNR = TN / (TN + FP)

    # False positive rate
    FPR = FP / (FP + TN)

    # return df_ious['iou'].mean()
    return accuracy_score(y_true_list, y_pred_list), df_ious['iou'].mean(), np.mean(FPR)

if __name__ == '__main__':

    yoloversion = 'yolov7'
    all_results = glob.glob(r'F:\202105_PAI\data\P1_results\yolov7_img640_b8_e300_hyp_custom\*')
    yolo_results = [f for f in all_results if yoloversion in f]
    yolo_results = [f for f in yolo_results if 'conf' in f]
    mean_iou_list = []
    mean_oa_list = []
    FPR_list = []
    for data_i, data_path in enumerate(yolo_results):
        print('{} of {} datasets'.format(data_i, len(yolo_results)))
        if not yoloversion == 'yolov4':
            data_path = os.path.join(data_path, 'labels')
        mean_oa, mean_iou, FPR = run_evaluate(data_path=data_path, plot_cm=False)
        mean_iou_list.append(mean_iou)
        mean_oa_list.append(mean_oa)
        FPR_list.append(FPR)

    iou_array = np.reshape(np.array(mean_iou_list), (-1, 9))
    oa_array = np.reshape(np.array(mean_oa_list), (-1, 9))
    FPR_array = np.reshape(np.array(FPR_list), (-1, 9))

    metric_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    conf_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    cm = pd.DataFrame(iou_array, index=conf_threshold, columns=metric_threshold)
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt='.4f', ax=ax, cmap='Blues', square=True, vmin=0, vmax=1)
    ax.set_xlabel('\nIoU Threshold')
    ax.set_ylabel('Confidence Threshold\n')
    ax.figure.tight_layout()
    ax.figure.subplots_adjust(bottom=0.2)
    ax.invert_yaxis()
    plt.title('IoU')
    plt.savefig(r'F:\202105_PAI\data\P1_results\thresholds_' + yoloversion + '_iou.png')
    plt.show()

    cm = pd.DataFrame(oa_array, index=conf_threshold, columns=metric_threshold)
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt='.4f', ax=ax, cmap='Blues', square=True, vmin=0, vmax=1)
    ax.set_xlabel('\nIoU Threshold')
    ax.set_ylabel('Confidence Threshold\n')
    ax.figure.tight_layout()
    ax.figure.subplots_adjust(bottom=0.2)
    ax.invert_yaxis()
    plt.title('Overall Accuracy')
    plt.savefig(r'F:\202105_PAI\data\P1_results\thresholds_' + yoloversion + '_oa.png')
    plt.show()

    cm = pd.DataFrame(FPR_array, index=conf_threshold, columns=metric_threshold)
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt='.4f', ax=ax, cmap='Blues', square=True, vmin=0, vmax=1)
    ax.set_xlabel('\nIoU Threshold')
    ax.set_ylabel('Confidence Threshold\n')
    ax.figure.tight_layout()
    ax.figure.subplots_adjust(bottom=0.2)
    ax.invert_yaxis()
    plt.title('False Positive Rate')
    plt.savefig(r'F:\202105_PAI\data\P1_results\thresholds_' + yoloversion + '_FPR.png')
    plt.show()


    print('finished')
