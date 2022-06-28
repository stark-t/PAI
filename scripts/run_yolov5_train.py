# import packages
import os

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

    # Maybe check if yolo is there?

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'relative/path/to/file/you/want')

    if not os.path.exists(r'.code\yolov5\train.py'):
        d=1

    train.run(data=data_yaml, weights=weights, batch_size=batch_size,
              epochs=epochs, imgsz=image_size,hyp=hyperparms, project=save_dir)

    print('finished')
