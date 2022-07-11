# import packages
import os
import glob
import pandas as pd
import yaml
# import scripts


def file_names_func(path):
    """

    :param path: path to  data
    :return: list of names
    """

    file_names = [f.split(os.sep)[-1] for f in path]
    file_names = [f.split('.')[0] for f in file_names]

    return file_names

def get_datapath_func(data_path, verbose=2):
    """

    :param data_path: path to original data
    :param verbose: select print level
    :return: dataframe of all images and labels per class
    """
    data_dirs = glob.glob(os.path.join(data_path, '*'))
    data_dirs = [f for f in data_dirs if 'img' in f]

    dfs = []

    # loop through each data folder
    for data_dir_i, data_dir in enumerate(data_dirs):
        class_name = data_dir.split(os.sep)[-1]
        class_name = class_name.split('_sample')[0]
        class_name = class_name.split('img_')[1]
        class_name = class_name.title()
        # get all images
        images_path = glob.glob(os.path.join(data_dir, 'img', '*'))
        labels_path = glob.glob(os.path.join(data_dir, 'annotations', 'yolo_txt', '*'))

        # get id file names
        images_file_names = file_names_func(images_path)
        label_file_names = file_names_func(labels_path)

        # create dataframes for images and labels
        image_zip = list(zip(images_file_names, images_path))
        image_df = pd.DataFrame(image_zip, columns=['file_names', 'images_path'])
        image_df['class'] = pd.Series(str(class_name), index=image_df.index, dtype='category')

        labels_zip = list(zip(label_file_names, labels_path))
        label_df = pd.DataFrame(labels_zip, columns=['file_names', 'labels_path'])

        class_df = pd.merge(image_df, label_df, on='file_names', how='outer')
        dfs.append(class_df)

    # merge list of dfs
    df = pd.concat(dfs)

    if verbose >= 2:
        print('Original dataset')
        print_df = df.groupby(['class'])['images_path', 'labels_path'].count()
        print(print_df)

    return df


if __name__ == '__main__':

    # get current dir
    dirname = os.path.dirname(__file__)

    # read yaml config file
    data_yaml = os.path.join(dirname, 'config_yolov5.yaml')
    with open(data_yaml) as file:
        data = yaml.safe_load(file)

    images_labels_df = get_datapath_func(data_path=data['data_path'], verbose=data['verbose'])

    print('finished')
