# import packages
import os
from git import Repo
import yaml
import sys

# import scripts
from utils_create_datasets import run_create_datasets

if __name__ == '__main__':

    # get current dir
    dirname = os.path.dirname(__file__)

    # read yaml config file
    data_yaml = os.path.join(dirname, 'config_yolov5.yaml')
    try:
        with open(data_yaml) as file:
            data = yaml.safe_load(file)
    except:
        print('Create config file for YOLOv4')
        sys.exit(1)

    # check if yolo needs to be downloaded
    # split and create yolov5 dir (should be os independed)
    dirname_splits = os.path.normpath(dirname).split(os.path.sep)[0:-1]
    drive_path = dirname_splits[0] + os.sep
    detectors_path = os.path.join(drive_path, *dirname_splits[1:])
    detectors_path = os.path.join(detectors_path, 'detectors', 'yolov5')
    detectors_path_yolov5_trainpy = os.path.join(detectors_path, 'train.py')
    # check if train.py exists
    if not os.path.isfile(detectors_path_yolov5_trainpy):
        print('Start downloading yolov5 from github into {}'.format(detectors_path))
        Repo.clone_from('https://github.com/ultralytics/yolov5.git', detectors_path)
        print('Finished cloning')

    # check if dataset exists
    if not os.path.isdir(data['train']):
        run_create_datasets()

    print('finished')
