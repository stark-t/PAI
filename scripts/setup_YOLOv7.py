# import packages
import os
from git import Repo
import yaml
import sys
import subprocess

# import scripts
from utils_create_datasets import run_create_datasets

if __name__ == '__main__':

    # get current dir
    dirname = os.path.dirname(__file__)

    # read yaml config file
    data_yaml = os.path.join(dirname, 'config.yaml')
    try:
        with open(data_yaml) as file:
            data = yaml.safe_load(file)
    except:
        print('Create config file')
        sys.exit(1)

    # check if yolo needs to be downloaded
    # split and create yolov5 dir (should be os independed)
    dirname_splits = os.path.normpath(dirname).split(os.path.sep)[0:-1]
    drive_path = dirname_splits[0] + os.sep
    detectors_path = os.path.join(drive_path, *dirname_splits[1:])
    detectors_path = os.path.join(detectors_path, 'detectors', 'yolov7')
    detectors_path_yolov7_trainpy = os.path.join(detectors_path, 'train.py')
    # check if train.py exists
    if not os.path.isfile(detectors_path_yolov7_trainpy):
        print('Start downloading yolov7 from github into {}'.format(detectors_path))
        Repo.clone_from('https://github.com/WongKinYiu/yolov7.git', detectors_path)
        print('Finished cloning')

    # check if dataset exists
    if not os.path.isdir(data['train']):
        run_create_datasets()

    # create save and run dir
    if not os.path.exists(data['runsave_basepath']):
        os.makedirs(data['runsave_basepath'])

    if os.path.isfile(detectors_path_yolov7_trainpy):
        if data['run_within_setupfile']:
            subprocess.call([sys.executable,
                             "C://Users//star_th//PycharmProjects//PAI//detectors//yolov7//train.py",
                             "--weights",
                             "F://202105_PAI//data//P1_training_runs//yolov7//weights//yolov7.pt",
                             "--data", "C://Users//star_th//PycharmProjects//PAI//scripts//config.yaml",
                             "--hyp",
                             "C://Users//star_th//PycharmProjects//PAI//detectors//yolov7//data//hyp.scratch.tiny.yaml",
                             "--device", "0",
                             "--batch-size", str(data['batch_size']),
                             "--epochs", str(data['epochs']),
                             "--img", "1280",
                             "--project", str(data['runsave_basepath']),
                             "--name", 'yolov7',
                             "--exist-ok"],
                            shell=False)
        else:
            print('Cannot run training within setup!')
            print('Run file again to refresh setup.')
            sys.exit(1)

    print('finished')
