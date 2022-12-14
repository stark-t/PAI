# import packages
import os
from os.path import exists as file_exists
import glob
import shutil

import pandas as pd
import numpy as np
import torch
import tqdm as t
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from sklearn.metrics import accuracy_score
from numpy import unravel_index
import math
from shutil import copyfile

# import scripts
# from detectors.yolov5.utils.metrics import bbox_iou
from scripts.utils_confusionmatrix import cm_analysis
from scripts.plot_predlabel import plot_labelprediction
from scripts.utils_getfileinfo import get_file_info

def get_ytrue_y_pred(df):
    y_true_list = list(df['class_label'])
    y_true_list = [int(f) + 1 if f >= 0 else 0 for f in y_true_list]

    y_pred_list = list(df['class_pred'])
    # y_pred_list = [int(f) + 1 if f >= 0 else 0 for f in y_pred_list]
    y_pred_list = [int(f) + 1 if f >= 0 else int(f) for f in y_pred_list]

    y_true_list[0] = -1
    y_true_list[1] = 0
    y_pred_list[0] = 0
    y_pred_list[1] = -1
    y_pred_unique = np.unique(y_pred_list, return_counts=True)
    y_true_unique = np.unique(y_true_list, return_counts=True)

    return y_true_list, y_pred_list, y_pred_unique

def accuracy_metrics(df, df_name='all'):
    y_true_list, y_pred_list, y_pred_unique = get_ytrue_y_pred(df)

    labels = ['Background_FP', 'Background', 'Araneae', 'Coleoptera', 'Diptera', 'Hemiptera',
              'Hymenoptera_Formicidae', 'Hymenoptera', 'Lepidoptera', 'Orthoptera']

    labels_i = list(range(len(labels)))
    label_pred_i = list(y_pred_unique[0] + 1)
    label_match = list(set(labels_i).intersection(label_pred_i))
    labels = [labels[i] for i in label_match]
    labels = [f for fi, f in enumerate(label_pred_i) if 1 == 1]
    if df_name == 'df_mutliples':
        d=1
    # create path to save metrics, confusion matrix, etc
    results_path = data_path.split(os.sep)
    results_path = os.path.join(*results_path[:-1])
    filename = os.path.join(results_path, 'confusion_matrix.png')
    filename_tex = os.path.join(results_path, 'confusion_matrix_tex.txt')
    # df_ious_groupPercentile.to_csv(os.path.join(results_path, 'class_ious_percentiles.csv'))

    cm = cm_analysis(y_true_list, y_pred_list, labels, ymap=None, figsize=(9, 9),
                     filename=filename, filename_tex=None, plot=None)

    # from confusion matrix create metrics
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Accuarcy
    OA = np.divide((TN + TP), (TN + FP + TP + FN + 1e-5))
    OA = [np.nan if x == 0 else x for x in OA]
    # False positive rate
    FPR = np.divide(FP, (FP + TN + 1e-5))
    FPR = [np.nan if x == 0 else x for x in FPR]
    # Precision
    P = np.divide(TP, (TP + FP + 1e-5))
    P = [np.nan if x == 0 else x for x in P]
    # Recall
    R = np.divide(TP, (TP + FN + 1e-5))
    R = [np.nan if x == 0 else x for x in R]

    IoU = df['iou'].mean()

    matching_ids = []
    for i in range(len(y_true_list)):
        if y_true_list[i] == y_pred_list[i]:
            matching_ids.append(i)

    df_ious_matching_classes = df.iloc[matching_ids]
    IoU_match = df_ious_matching_classes['iou'].mean()

    filename_metrics = filename_tex.replace('confusion_matrix_tex', ('metrics_') + df_name)
    with open(filename_metrics, 'w') as f:
        f.write('OA: {:6.4f}, FPR: {:6.4f}, Precision: {:6.4f}, Recall: {:6.4f}, IoU_all: {:6.4f}, IoU_match: {:6.4f}'.
                format(
            np.nanmean(OA),
            np.nanmean(FPR),
            np.nanmean(P),
            np.nanmean(R),
            IoU,
            IoU_match,
        ))

    # Print metrics
    print('Metrics for: {}'.format(df_name))
    print('Accuracy: {:.4f}'.format(np.nanmean(OA)))
    print('FPR: {:.4f}'.format(np.nanmean(FPR)))
    print('Precision: {:.4f}'.format(np.nanmean(P)))
    print('Recall: {:.4f}'.format(np.nanmean(R)))
    print('{} images with an IoU of: {:.4f}'.format(df.shape[0], IoU))
    print('{} images with matching classes and an IoU of: {:.4f}'.format(df_ious_matching_classes.shape[0], IoU_match))
    print('\n')

    return OA, P, R, FPR, IoU, IoU_match, cm



def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def calculate_ious(series, valid_labels=1):
    # claculate iou, class doenst factor in!
    # input: pair label(s) and prediction(s) from one image/label file
    ious = []
    records = []
    # 1st go through each label
    for i in range(len(series['bbox_label'])):
        # 2nd go through each prediction
        for j in range(len(series['bbox_pred'])):

            # convert label bbox from numpy to pytoch tensor
            bbox_label = np.array(series['bbox_label'][i]).squeeze()
            bbox_label_torch = torch.from_numpy(bbox_label)
            bbox_label_torch = bbox_label_torch[None, :]

            # convert prediction bbox from numpy to pytoch tensor
            bbox_prediction = np.array(series['bbox_pred'][j]).squeeze()
            bbox_prediction_torch = torch.from_numpy(bbox_prediction)
            bbox_prediction_torch = bbox_prediction_torch[None, :]

            # check if classes match, else set label and bbox to false
            if series['class_label'][i] != series['class_pred'][j] and \
                    series['class_label'][i] >= 0 and series['class_pred'][j] >= 0:
                iou = -1.0
            # if there is a dummy predcition or label present, set iou to -2
            elif series['class_label'][i] < 0 or series['class_pred'][j] < 0:
                iou = -2.0
            # calculate actual iou
            else:
                iou_torch = bbox_iou(bbox_label_torch, bbox_prediction_torch)
                iou = iou_torch.numpy().squeeze()
            # append iou to list
            ious.append(iou)
            # create a record with all necessary informations
            record = {
                'ID': series.ID,
                'file_pred': series.file_pred,
                'class_pred': series.class_pred[j],
                'bbox_pred': series.bbox_pred[j],
                'file_label': series.file_label,
                'class_label': series.class_label[i],
                'bbox_label': series.bbox_label[i],
                'iou': iou,
                'n_BB_labels': series.n_BB_labels,
            }
            records.append(record)

    # create dataframe from all possible iou combinations
    df = pd.DataFrame(records)

    # in case of multiple labels and predcitions all iou combinations have been calculated
    # only select highest lable prediction pair with the highest iou
    # 1st sort indicies based on iou value
    indicies_iou_sort_arr = np.argsort(ious)
    # 2nd create a list of sorted indicies
    indicies_iou_sort_list = list(indicies_iou_sort_arr)
    # 3rd reverse sorted list so the indicies of the highest ious are on top
    indicies_iou_sort_list = list(reversed(indicies_iou_sort_list))
    # 4th get the first n results from label prediction pair
    # n = sqrt(length of all possible label prediction pairs)
    # e.g 2 labels 2 prediction = 4 iou pairs --> n = sqrt(4) = 2 ious
    indicies_iou = indicies_iou_sort_list[0:int(np.sqrt(len(df)))]
    # get the actual iou pair by their indicies
    df_iou_pairs = df.iloc[indicies_iou, :]

    # change dataframe items to get true class labels corresponding to wrong predictions
    # set iou and class_pred to -1 if valid_predictions dont fit
    df_iou_pairs.reset_index(drop=True, inplace=True)
    for index, row in df_iou_pairs.iterrows():
        if index > valid_labels -1:
            df_iou_pairs.at[index, 'class_pred'] = -1.0
            df_iou_pairs.at[index, 'class_label'] = -1.0
            df_iou_pairs.at[index, 'iou'] = np.NaN
            df_iou_pairs.at[index, 'n_BB_labels'] = np.NaN

    return df_iou_pairs


def run_evaluate(data_path='1', plot_cm=False):
    """

    :param input:
    :return:
    """

    # get all prediction files in data_path
    predictions_list = glob.glob(data_path + os.sep + '*.txt')
    # if there are any predictions in the threshold folder or not
    if len(predictions_list) > 0:
        # run get_file_info function if there are predictions and read prediction files to create a dataframe
        records = get_file_info(predictions_list)
        df_predictions = pd.DataFrame(records)
    else:
        # if there are no predictions return an empty dataframe
        df_predictions = pd.DataFrame(columns=['ID', 'file', 'class', 'bbox'])

    # get all label files from a fixed path and create label dataframe
    if not 'syrphidae' in data_path:
        labels_list = glob.glob(r'F:\202105_PAI\data\P1_Data_sampled\test_valentin\labels' + os.sep + '*')
        records = get_file_info(labels_list)
        df_labels = pd.DataFrame(records)
    else:
        labels_list = glob.glob(r'F:\202105_PAI\data\P1_results\img_syrphidae_sample_2022_06_17_annotated\labels' +
                                os.sep + '*')
        records = get_file_info(labels_list)
        df_labels = pd.DataFrame(records)
        for index, row in df_labels.iterrows():
            class_label_list = []
            for l in range(len(row['bbox'])):
                class_label_list.append(2.0)
            df_labels.at[index, 'class'] = class_label_list
            # df_labels.at[index, 'class'] = [2.0]

    # merge label and prediction dataframe
    # merge on ID (file name) if there is no matching label/prediction a nan row will be added
    df = pd.merge(df_predictions, df_labels, on='ID', how="outer", suffixes=('_pred', '_label'))

    # replace nan in dataframe for labels or predictions with dummy values so ious can be calculated
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        if not isinstance(row['class_pred'], list):
            df.at[index, 'class_pred'] = [-1.0]
            df.at[index, 'bbox_pred'] = [[.0, .0, .0, .0]]
        if not isinstance(row['class_label'], list):
            df.at[index, 'class_label'] = [-1.0]
            df.at[index, 'bbox_label'] = [[.0, .0, .0, .0]]

    # add number of lable bounding boxes into new column
    for index, row in df.iterrows():
        n_BB_labels = len(row['class_label'])
        if n_BB_labels == 1:
            class_n_BB_labels = 1
        else:
            class_n_BB_labels = 2

        df.at[index, 'n_BB_labels'] = class_n_BB_labels

    # loop through each label and prediction pair
    # if there is a missmatch of numbers ad a dummy value so iou can be calculated
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        # check if there are less labels than predictions
        if len(row['class_label']) < len(row['class_pred']):
            # calculate the difference between number of labels and predictions
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

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        if len(row['class_label']) > 1 and row['class_label'] != row['class_pred']:
            print(index, row['class_label'], row['class_pred'])

    # calclulate iou pairs
    records = []
    # loop through each prediction and label pair to calculate iou
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        if index == 187:
            d=1
        # get number of true labels
        n_valid_labels = [f for f in row['class_label'] if f >= 0]
        n_valid_labels = len(n_valid_labels)
        df_iou_combination = calculate_ious(row, valid_labels=n_valid_labels)
        # for each returned iou(s) add single iou to a list
        for _, record in df_iou_combination.iterrows():
            records.append(record.to_dict())

    # calculate IoU percentiles
    df_ious = pd.DataFrame(records)
    df_ious_groupPercentile = df_ious.groupby('class_label')['iou'].agg([percentile(1), percentile(25), percentile(50),
                                                                         percentile(75), percentile(99)])

    test_hymenoptera = False
    if test_hymenoptera:
        hymenoptera_df = df_ious.loc[((df_ious['class_label'] == 5) & (df_ious['iou'] > .95))]
        for index, row in hymenoptera_df.iterrows():
            image_path_file_ID = row['ID']
            image_path_file = glob.glob(r'F:\202105_PAI\data\P1_Data' + os.sep + '**' + os.sep + 'img' + os.sep + image_path_file_ID + '*')[0]
            label_file = row['file_label']
            prediction_file = row['file_pred']

            if os.path.isfile(image_path_file) and \
                    os.path.isfile(label_file) and \
                    os.path.isfile(prediction_file):
                plot_labelprediction(path_file=image_path_file, label_file=label_file, prediction_file=prediction_file,
                                     plot_show=False)

    test_syrphidae = False
    if test_syrphidae:
        hymenoptera_df = df_ious.loc[((df_ious['class_label'] == 2) & (df_ious['iou'] > .95))]
        for index, row in hymenoptera_df.iterrows():
            image_path_file_ID = row['ID']
            image_path_file = glob.glob(r'F:\202105_PAI\data\P1_Data' + os.sep + '**' + os.sep + 'img' + os.sep + image_path_file_ID + '*')[0]
            label_file = row['file_label']
            prediction_file = row['file_pred']

            if os.path.isfile(image_path_file) and \
                    os.path.isfile(label_file) and \
                    os.path.isfile(prediction_file):
                plot_labelprediction(path_file=image_path_file, label_file=label_file, prediction_file=prediction_file,
                                     plot_show=False)

    # test_multiple = True
    # if test_multiple:
    #     hymenoptera_df = df_ious.loc[((df_ious['class_label'] == 2) & (df_ious['iou'] > .8) & (df_ious['n_BB_labels'] > 1))]
    #     for index, row in hymenoptera_df.iterrows():
    #         image_path_file = row['ID']
    #         image_path_file = os.path.join(r'F:\202105_PAI\data\P1_Data\img_diptera_sample_2021_09_20\img',
    #                                        (image_path_file + '.jpg'))
    #         label_file = row['file_label']
    #         prediction_file = row['file_pred']
    #
    #         if os.path.isfile(image_path_file) and \
    #                 os.path.isfile(label_file) and \
    #                 os.path.isfile(prediction_file):
    #             plot_labelprediction(path_file=image_path_file, label_file=label_file, prediction_file=prediction_file,
    #                                  plot_show=False)

    test_multiple = True
    if test_multiple:
        # df_m = df_ious.loc[df_ious['n_BB_labels'] > 1]
        df_m = df_ious.loc[df_ious['iou'] > .8]
        for index, row in df_m.iterrows():
            image_path_file_ID = row['ID']
            image_path_file = glob.glob(r'F:\202105_PAI\data\P1_Data' + os.sep + '**' +
                                        os.sep + 'img' + os.sep + image_path_file_ID + '*')[0]
            label_file = row['file_label']
            prediction_file = row['file_pred']

            if os.path.isfile(image_path_file) and \
                    os.path.isfile(label_file) and \
                    os.path.isfile(prediction_file):
                plot_labelprediction(path_file=image_path_file, label_file=label_file, prediction_file=prediction_file,
                                     plot_show=False)

    OA, P, R, FPR, IoU, IoU_match, cm = accuracy_metrics(
        df_ious, df_name='df_all')
    df_singels = df_ious.loc[df_ious['n_BB_labels'] == 1]
    OA_singels, P_singels, R_singels, FPR_singels, IoU_singels, IoU_match_singels, cm = accuracy_metrics(
        df_singels, df_name='df_singels')
    df_mutliples = df_ious.loc[df_ious['n_BB_labels'] == 2]
    OA_mutliples, P_mutliples, R_mutliples, FPR_mutliples, IoU_mutliples, IoU_match_mutliples, cm = accuracy_metrics(
        df_mutliples, df_name='df_mutliples')

    if 'syrphidae' in data_path:
        df_syrphidae = df_ious.loc[df_ious['class_label'] == 2]
        OA_syrphidae, P_syrphidae, R_syrphidae, FPR_syrphidae, IoU_syrphidae, IoU_match_syrphidae, cm = accuracy_metrics(
            df_syrphidae, df_name='df_syrphidae')

    if not 'syrphidae' in data_path:
        df_hymenoptera = df_ious.loc[df_ious['class_label'] == 5]
        OA_hymenoptera, P_hymenoptera, R_hymenoptera, FPR_hymenoptera, IoU_hymenoptera, IoU_match_hymenoptera, cm = accuracy_metrics(
            df_hymenoptera, df_name='df_hymenoptera')

        df_diptera = df_ious.loc[df_ious['class_label'] == 2]
        OA_diptera, P_diptera, R_diptera, FPR_diptera, IoU_diptera, IoU_match_diptera, cm = accuracy_metrics(
            df_diptera, df_name='df_diptera')

    return OA, IoU_match, FPR, P, R

if __name__ == '__main__':
    # select path to results
    # list all file in path directory
    # source_path = r'F:\202105_PAI\data\P1_results\yolov5_n_img640_b8_e300_hyp_custom\results_at_conf_0.3_iou_0.1'
    # source_path = r'F:\202105_PAI\data\P1_results\yolov5_s_img640_b8_e300_hyp_custom\results_at_conf_0.3_iou_0.1'
    source_path = r'F:\202105_PAI\data\P1_results\yolov7_tiny_img640_b8_e300_hyp_custom\results_at_conf_0.3_iou_0.1'
    # source_path = r'F:\202105_PAI\data\P1_results\job_191869_syrphidae_loop_detect_with_191623_yolov7_tiny_img640_b8_e300_hyp_custom\results_at_conf_0.3_iou_0.1'
    # source_path = r'F:\202105_PAI\data\P1_results\yolov5_n_img640_b8_e300_hyp_custom'


    all_results = glob.glob(source_path + '\*')
    # select yolo-version for naming and searching for labels since process for v4 is different than v5 and v7
    yoloversion = source_path.split(os.sep)[-2]
    if not 'syrphidae_loop' in yoloversion:
        yoloversion = yoloversion.split('_')[0:2]
        yoloversion = '_'.join(yoloversion)
    else:
        yoloversion = 'yolov7_tiny_syrphidae'

    if 'results_at' in source_path:
        # only select path if it is a folder
        yolo_results = [f for f in all_results if os.path.isdir(f)]
        yolo_results = yolo_results[0].split(os.sep)
        yolo_results = os.path.join(*yolo_results[:])
        yolo_results = [yolo_results]
    else:
        yolo_results = [f for f in all_results if os.path.isdir(f)]

    # create empty lists for accuarcy metrics
    mean_iou_list = []
    mean_oa_list = []
    FPR_list = []
    P_list = []
    R_list = []
    FP_list = []

    # go through each folder and calculate accuracy metrics and append it to lists
    for data_i, data_path in enumerate(yolo_results):
        print('{} of {} datasets'.format(data_i, len(yolo_results)))
        mean_oa, mean_iou, FPR, P, R = run_evaluate(data_path=data_path, plot_cm=False)
        mean_iou_list.append(mean_iou)
        mean_oa_list.append(mean_oa)
        FPR_list.append(FPR)
        P_list.append(P)
        R_list.append(R)

    if len(mean_oa_list) > 1:
        # Turn list of all 81 (9*9) results into 9*9 array
        iou_array = np.reshape(np.array(mean_iou_list), (-1, 9))
        oa_array = np.reshape(np.array(mean_oa_list), (-1, 9))
        FPR_array = np.reshape(np.array(FPR_list), (-1, 9))
        P_array = np.reshape(np.array(P_list), (-1, 9))
        R_array = np.reshape(np.array(R_list), (-1, 9))
        FP_array = np.reshape(np.array(FP_list), (-1, 9))

        # hardcode labels for thresholds
        metric_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        conf_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        low_FPR = unravel_index(FPR_array.argmin(), FPR_array.shape)
        print('Lowest FPR at confidence {:.1f} and iou {:.1f} with {:.4f}'.format(conf_threshold[low_FPR[0]], metric_threshold[low_FPR[1]], FPR_array[low_FPR[0], low_FPR[1]]))

        # max_P = unravel_index(P_array.argmax(), P_array.shape)
        # print('Highest precission at confidence {:.1f} and iou {:.1f} with {:.4f}'.format(conf_threshold[max_P[0]], metric_threshold[max_P[1]], iou_array[max_P[0], max_P[1]]))
        #
        # max_R = unravel_index(R_array.argmax(), R_array.shape)
        # print('Highest recall at confidence {:.1f} and iou {:.1f} with {:.4f}'.format(conf_threshold[max_R[0]], metric_threshold[max_R[1]], iou_array[max_R[0], max_R[1]]))

        max_oa = unravel_index(oa_array.argmax(), oa_array.shape)
        print('Highest OA at confidence {:.1f} and iou {:.1f} with {:.4f}'.format(conf_threshold[max_oa[0]], metric_threshold[max_oa[1]], oa_array[max_oa[0], max_oa[1]]))

        low_FP = unravel_index(FP_array.argmin(), FP_array.shape)
        print('Lowest FP at {:.1f} and iou {:.1f}'.format(conf_threshold[low_FP[0]], metric_threshold[low_FP[1]]))

        src_file = source_path + r'\results_at_conf_' + \
                str(conf_threshold[max_oa[0]]) + '_iou_' + \
                str(conf_threshold[max_oa[1]]) + '\metrics.txt'

        dst_file = source_path + r'\best_metricsat_conf_' + \
                str(conf_threshold[max_oa[0]]) + '_iou_' + \
                str(conf_threshold[max_oa[1]]) + '.txt'
        copyfile(src_file, dst_file)

        # create heatmaps for all accuracy metrics
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
        sns.heatmap(cm, annot=True, fmt='.4f', ax=ax, cmap='Blues', square=True, vmin=0, vmax=cm.max())
        ax.set_xlabel('\nIoU Threshold')
        ax.set_ylabel('Confidence Threshold\n')
        ax.figure.tight_layout()
        ax.figure.subplots_adjust(bottom=0.2)
        ax.invert_yaxis()
        plt.title('False Positive Rate')
        plt.savefig(r'F:\202105_PAI\data\P1_results\thresholds_' + yoloversion + '_FPR.png')
        plt.show()


    print('finished')
