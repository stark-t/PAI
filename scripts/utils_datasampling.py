# import packages
import os
import pandas as pd
import numpy as np
import random

# import scripts
import utils_config as config
from utils_datapaths import get_datapath_func

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_balanced_testval_df(df, min_count_testval):
    """

    """
    group_classes = df.groupby(['class'])
    indices_testval = list(
        np.hstack([np.random.choice(v, min_count_testval, replace=False) for v in group_classes.groups.values()]))
    df_sampled = df.loc[indices_testval]

    return df_sampled

def datasampling_func(df=pd.DataFrame(), traintestval_ratio=[.7, .15, .15], verbose=2):
    """

    :param df:
    :return:
    """

    set_seed(config.seed)

    # select if images without labels should come into training, testing, validation dataset
    if not config.no_labels:
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)

    group_classes = df.groupby(['class'])
    min_count = group_classes['labels_path'].count().min()

    df_test = get_balanced_testval_df(df, int(min_count * config.traintestval_ratio[1]))
    if config.verbose >= 2:
        print('\nNumber of image tiles per class in {}% valdiation dataset'.format(config.traintestval_ratio[1]*100))
        print_df = df_test.groupby(['class'])['images_path', 'labels_path'].count()
        print(print_df)

    # remove test entries from dataframe for training and validation split
    df_x = pd.concat([df_test, df])
    df_trainval = df_x.drop_duplicates(keep=False)
    df_trainval.reset_index(drop=True, inplace=True)
    del df_x

    df_val = get_balanced_testval_df(df_trainval, int(min_count * config.traintestval_ratio[2]))
    if config.verbose >= 2:
        print('\nNumber of image tiles per class in {}% valdiation dataset'.format(config.traintestval_ratio[2]*100))
        print_df = df_val.groupby(['class'])['images_path', 'labels_path'].count()
        print(print_df)

    df_x = pd.concat([df_test, df_val, df])
    df_train = df_x.drop_duplicates(keep=False)
    del df_x
    if config.verbose >= 2:
        print('\nNumber of image tiles per class in training dataset')
        print_df = df_train.groupby(['class'])['images_path', 'labels_path'].count()
        print(print_df)

    return df_train, df_test, df_val


if __name__ == '__main__':
    df = get_datapath_func(data_path=config.data_path, verbose=config.verbose)
    df_train, df_test, df_val = datasampling_func(df=df, traintestval_ratio=config.traintestval_ratio,
                                                  verbose=config.verbose)

    print('finished')
