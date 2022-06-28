# import packages
import os
from git import Repo

# import scripts
import utils_config as config
from detectors.yolov5 import train


if __name__ == '__main__':

    data_yaml = r'C:\Users\star_th\PycharmProjects\PAI\scripts\P1_data.yaml'
    weights = 'yolov5s6.pt'
    epochs = 3
    batch_size = 8
    image_size = 1280
    hyperparms = r'C:\Users\star_th\PycharmProjects\PAI\code\yolov5\data\hyps\hyp.scratch-med.yaml'
    save_dir = r'F:\202105_PAI\data\P1_yolov5'

    # check if yolo needs to be downloaded
    # get current dir
    dirname = os.path.dirname(__file__)
    # split and create yolov5 dir
    dirname_splits = os.path.normpath(dirname).split(os.path.sep)[0:-1]
    drive_path = dirname_splits[0] + os.sep
    detectors_path = os.path.join(drive_path, *dirname_splits[1:])
    detectors_path_yolov5 = os.path.join(detectors_path, 'detectors', 'yolov5')
    detectors_path_yolov5_trainpy = os.path.join(detectors_path, 'train.py')
    # check if train.py exists
    if not os.path.isfile(detectors_path_yolov5_trainpy):
        print('download yolov5 into /PAI/detectors/yolov5')
        print('download yolov5 from github')
        Repo.clone_from('https://github.com/ultralytics/yolov5.git', detectors_path)


    train.run(data=data_yaml, weights=weights, batch_size=batch_size,
              epochs=epochs, imgsz=image_size,hyp=hyperparms, project=save_dir)

    print('finished')
