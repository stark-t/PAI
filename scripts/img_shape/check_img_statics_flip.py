# Script to check if images were flipped by the VGG VIA annotation tool and flip back their width & height to
# comply with PIL image width & height reads (which is used in YOLOv5).

import pandas as pd
import os
import yaml
import glob
# pip install opencv-python # install OpenCV in terminal, in the PAI environment
import cv2
from utils_annotation import *
from scripts.utils_datapaths import *

if __name__ == '__main__':

    # Get current dir
    dirname = os.path.join(os.path.dirname( __file__ ), '..')
    # dirname = '/home/vs66tavy/PAI/scripts'
    print('dirname:', dirname)

    # Read yaml config file
    data_yaml = os.path.join(dirname, 'config_yolov5.yaml')
    with open(data_yaml) as file:
        data = yaml.safe_load(file)

    # Read all image paths with the helper function get_datapath_func() from utils_datapaths.py
    images_labels_df = get_datapath_func(data_path=data['data_path'], verbose=1)
    # Get file name from the path so that we can join later. os.path.split(path)[-1] splits a path based on OS path
    # separator and extracts the last component which is the file name + its extension:
    images_labels_df['file_name'] = [os.path.split(path)[-1] for path in images_labels_df['images_path']]

    # Read image attributes using PIL based on the image paths read above
    df_pil = get_img_width_height_pil_from_list(images_labels_df['images_path'], get_orientation=True)
    # Get file name from the path so that we can join later with data frame from VIA.
    df_pil['file_name'] = [os.path.split(path)[-1] for path in df_pil['file_path']]
    # Write the data frame as CSV file
    # df_pil.to_csv('/home/vs66tavy/Nextcloud/insect-photos-url-gbif/gbif-occurences/P1_Data_internal/check_img_statics/pil_img_wh.csv', encoding='utf-8', index=False)

    # Read image attributes from VGG VIA json file (COCO format)
    json_dir = '/home/vs66tavy/Nextcloud/insect-photos-url-gbif/gbif-occurences/P1_Data_internal/coco'
    json_file_paths = glob.glob(json_dir + '/*.json')
    # For each json file, read the data in a data frame and then concatenate all dataframes
    dfs = []
    for json_path in json_file_paths:
        df_temp = get_img_width_height_from_via_coco(json_path)
        dfs.append(df_temp)
    df_via = pd.concat(dfs)

    # Merge image attribute data frames from PIL & VIA and check for width & height mismatches
    df = pd.merge(df_pil, df_via, on='file_name', how='inner')
    print(df_pil.shape[0] == df_via.shape[0]) # expect True. If not, then something went wrong.
    print(df_pil.shape[0] == df.shape[0]) # expect True here as well
    # Mark the width & height mismatches. Use either width or height, no need to test for both.
    df['dif'] = df['width_pil'] != df['width_via']

    # # Are there differences between VGG VIA & cv2? - No
    # # Read image width & height using cv2 - this is way slower than PIL because it reads each image as a numpy array
    # # in the RAM.
    # df_cv2 = get_img_width_height_cv2_from_list(images_labels_df['images_path'])
    # df2 = pd.merge(df, df_cv2, on='file_path', how='inner')
    # df2['dif2'] = df2['width_cv2'] != df2['width_via']
    # df2_test = df2[df2['dif2'] == True]
    # print(df2_test['file_name'])
    # # Just one case of a corrupted file
    # # (8967, 'Hemiptera_Rhyparochromidae_Taphropeltus_contractus_2837697319_826955.jpg')

    # Merge with images_labels_df so that we know which ones have annotation files.
    df = pd.merge(df, images_labels_df, on='file_name', how='inner')

    # Check the cases with mismatch.
    df.sort_values("file_name", inplace=True)
    df_check_all = df[df['dif'] == True]
    df_check_has_lbl = df_check_all[df_check_all['labels_path'].notnull()]
    print(df_check_has_lbl.shape[0])
    print(df_check_has_lbl[['file_name', 'width_via', 'width_pil']])
    # There are 23 cases that need visual inspection. Most probably we need to flip width with height
    # Write the data frame as CSV file
    df_check_has_lbl.to_csv('/home/vs66tavy/Nextcloud/insect-photos-url-gbif/gbif-occurences/P1_Data_internal/check_img_statics/df_check_has_lbl.csv',
                            encoding='utf-8',
                            index=False)