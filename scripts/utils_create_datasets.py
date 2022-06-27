# import packages
import os
import pandas as pd
import tqdm
import shutil

# import scripts
import utils_config as config
from utils_datasampling import datasampling_func
from utils_datapaths import get_datapath_func

def create_dataset_func(df, data_path):
    """

    :param df: input dataframe
    :return:
    """

    if not os.path.exists(data_path + '\\images'):
        os.makedirs(data_path + '\\images')
    if not os.path.exists(data_path + '\\labels'):
        os.makedirs(data_path + '\\labels')

    CLASSES = df['class'].unique().tolist()
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


if __name__ == '__main__':

    df = get_datapath_func(data_path=config.data_path, verbose=config.verbose)
    df_train, df_test, df_val = datasampling_func(df=df, traintestval_ratio=config.traintestval_ratio,
                                                  verbose=config.verbose)


    # make train, val and test into your dataset folder:
    train_PATH = config.data_path_sampled + '\\train'
    if not os.path.exists(train_PATH):
        os.makedirs(train_PATH)

    val_PATH = config.data_path_sampled + '\\val'
    if not os.path.exists(val_PATH):
        os.makedirs(val_PATH)

    test_PATH = config.data_path_sampled + '\\test'
    if not os.path.exists(test_PATH):
        os.makedirs(test_PATH)

    create_dataset_func(df=df_train, data_path=train_PATH)
    create_dataset_func(df=df_test, data_path=val_PATH)
    create_dataset_func(df=df_val, data_path=test_PATH)

    print('finished')
