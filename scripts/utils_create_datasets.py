# import packages
import os
import pandas as pd
import tqdm
import shutil
import yaml

# import scripts
from utils_datasampling import datasampling_func
from utils_datapaths import get_datapath_func

def create_dataset_func(df, data_path):
    """

    :param df: input dataframe
    :return:
    """
    # get current dir
    dirname = os.path.dirname(__file__)

    # read yaml config file
    data_yaml = os.path.join(dirname, 'config_yolov5.yaml')
    with open(data_yaml) as file:
        data = yaml.safe_load(file)

    if not os.path.exists(os.path.join(data_path, 'images')):
        os.makedirs(os.path.join(data_path, 'images'))
    if not os.path.exists(os.path.join(data_path, 'labels')):
        os.makedirs(os.path.join(data_path, 'labels'))

    CLASSES = data['names']
    for i, row in tqdm.tqdm(df.iterrows()):
        label_PATH_src = row['labels_path']
        image_PATH_src = row['images_path']
        file_name = row['file_names']
        class_name = row['class']
        class_id = CLASSES.index(class_name)

        label_PATH_dst = os.path.join(data_path, 'labels', (file_name + '.txt'))
        image_PATH_dst = os.path.join(data_path, 'images', (file_name + '.jpg'))

        # read label file and change classes if necessary
        if not os.path.exists(label_PATH_src):
            continue
        with open(label_PATH_src, 'r') as f:
            contents = f.readlines()
        f.close()
        with open(label_PATH_dst, 'w') as f:
            for line in contents:
                delimiter = ''
                line_string = delimiter.join(line)
                line_array = line_string.split()
                line_array[0] = class_id
                new_line = ' '.join(str(x) for x in line_array)
                new_line = new_line + '\n'
                f.write(new_line)
        f.close()

        # image_PATH_src = os.path.join(image_PATH, image_file)
        # image_PATH_dst = os.path.join(output_PATH, 'images', (label_name + '.jpg'))
        if os.path.exists(label_PATH_src):
            shutil.copyfile(image_PATH_src, image_PATH_dst)

    return 1

def run_create_datasets():

    # get current dir
    dirname = os.path.dirname(__file__)

    # read yaml config file
    data_yaml = os.path.join(dirname, 'config_yolov5.yaml')
    with open(data_yaml) as file:
        data = yaml.safe_load(file)

    df = get_datapath_func(data_path=data['data_path'], verbose=data['verbose'])
    df_train, df_test, df_val = datasampling_func(df=df, traintestval_ratio=data['traintestval_ratio'],
                                                  seed=data['seed'], verbose=data['verbose'])

    # make train, val and test into your dataset folder:
    train_PATH = os.path.join(str(data['path']), 'train')
    if not os.path.exists(train_PATH):
        os.makedirs(train_PATH)

    val_PATH = os.path.join(str(data['path']), 'val')
    if not os.path.exists(val_PATH):
        os.makedirs(val_PATH)

    test_PATH = os.path.join(str(data['path']), 'test')
    if not os.path.exists(test_PATH):
        os.makedirs(test_PATH)

    create_dataset_func(df=df_train, data_path=train_PATH)
    create_dataset_func(df=df_test, data_path=val_PATH)
    create_dataset_func(df=df_val, data_path=test_PATH)

if __name__ == '__main__':
    run_create_datasets()

    print('finished')
