# import packages
import os

# import scripts
import utils_config as config
from yolov5 import train

def run_training(data_yaml, weights, epochs, batch_size,
                 image_size, hyperparms, save_dir):
    """

    :param input:
    :return:
    """
    train.run(data=data_yaml, weights=weights, batch_size=batch_size,
              epochs=epochs, imgsz=image_size,hyp=hyperparms, project=save_dir)

    return 1


if __name__ == '__main__':

    data_yaml = r'C:\Users\star_th\PycharmProjects\PAI_P1\scripts\P1_data.yaml'
    weights = 'yolov5s6.pt'
    epochs = 3
    batch_size = 8
    image_size = 1280
    hyperparms = r'C:\Users\star_th\PycharmProjects\PAI_P1\yolov5\data\hyps\hyp.scratch-med.yaml'
    save_dir = r'F:\202105_PAI\data\P1_yolov5'

    # Maybe check if yolo is there?

    run_training(data_yaml=data_yaml, weights=weights, epochs=epochs, batch_size=batch_size,
                 image_size=image_size, hyperparms=hyperparms, save_dir=save_dir)

    print('finished')
