# import packages
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
import glob
# import scripts
from scripts.utils_getfileinfo import get_file_info

def yolo2rectangle(x, w=640, h=640):
    # Convert nx4 boxes from [x, y, w, h] normalized to [xupper, yleft, w, h]
    y = np.zeros(4)
    y[0] = int(w * (x[0] - x[2] / 2))  # top left x)
    y[1] = h * (x[1] - x[3] / 2)  # top left y
    y[2] = w * (x[2])  # w
    y[3] = h * (x[3])  # h

    return y


def plot_labelprediction(path_file=r'pathtofile', label_file=r'pathtofile', prediction_file=r'pathtofile', plot_show=True):
    """

    :param input:
    :return:
    """

    # read label and prediction files
    if type(label_file) != list:
        label_file = [label_file]
    label_records = get_file_info(label_file)
    if type(prediction_file) != list:
        prediction_file = [prediction_file]
    prediction_records = get_file_info(prediction_file)
    image = mpimg.imread(path_file)

    # get image ID and order and family name
    image_ID = path_file.split(os.sep)[-1]
    image_name_parts = image_ID.split('_')[:2]
    image_name = ' '.join(image_name_parts)
    # get path to save plot
    path_save_image_basepath = os.path.join(r'F:\202105_PAI\data\P1_figures', (image_name_parts[0] + '_predlabel'))
    if not os.path.isdir(path_save_image_basepath):
        os.makedirs(path_save_image_basepath)

    path_save_image = os.path.join(r'F:\202105_PAI\data\P1_figures', (image_name_parts[0] + '_predlabel'),
                                            image_ID)

    # get bounding box per file
    bbox_label = [label_records[0]['bbox'][fi] for fi, f in enumerate(label_records[0]['bbox'])]
    bbox_pred = [prediction_records[0]['bbox'][fi] for fi, f in enumerate(prediction_records[0]['bbox'])]
    # bbox_label = label_records[0]['bbox']
    # bbox_pred = prediction_records[0]['bbox']

    # conver bboxes to image pixels  xywhn2xyxy
    bbox_label_px_list = [yolo2rectangle(bbox_label[fi], w=image.shape[1], h=image.shape[0]) for fi, f in
                     enumerate(bbox_label)]
    bbox_pred_px_list = [yolo2rectangle(bbox_pred[fi], w=image.shape[1], h=image.shape[0]) for fi, f in
                     enumerate(bbox_pred)]
    # bbox_label_px = yolo2rectangle(bbox_label[0], w=image.shape[1], h=image.shape[0])  #bbox_label[0])
    # bbox_pred_px = yolo2rectangle(bbox_pred[0], w=image.shape[1], h=image.shape[0])  #bbox_label[0])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(image)
    for bbox_label_px in bbox_label_px_list:
        rect_labels = plt.Rectangle((bbox_label_px[0], bbox_label_px[1]), bbox_label_px[2],
                                  bbox_label_px[3], ec="red", fill=False)
        plt.gca().add_patch(rect_labels)

    for bbox_pred_px in bbox_pred_px_list:
        rect_preds = plt.Rectangle((bbox_pred_px[0], bbox_pred_px[1]), bbox_pred_px[2],
                                  bbox_pred_px[3], ec="blue", fill=False)
        plt.gca().add_patch(rect_preds)
    ax.set_title(image_name)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig(path_save_image)
    if plot_show:
        plt.show()
    else:
        plt.close()

    return 1


if __name__ == '__main__':

    print('finished')
