"""
Created on 22th of Nov. 2021
@author: Robin, Thomas

README:
- Read and change variables in: Config_placeholder.yaml in data folder of PAI project
- In the current file, change variables in lines 14-23
"""

import train
import os

if __name__ == "__main__":
    weights = 'yolov5m6.pt'
    # weights = r"C:\MASTERTHESIS\Results\Training\P1_beta_ID_100_yolov5m6\weights\best.pt"
    epochs = 3
    batch_size = 4
    image_size = 1280
    # hyp = r"C:\MASTERTHESIS\Results\Evolve\6days_6hours_nobeetles_nowandb\hyp_evolve.yaml"
    save_dir = r"C:\MASTERTHESIS\Results\Training"
    # config = 'Config_ArTaxOr_orders.yaml'
    # config = 'Config_ArTaxOr_ID.yaml'
    # config = 'Config_P1_beta_ID.yaml'
    # config = 'Config_P1_beta_orders.yaml'
    config = 'Config_P1_orders.yaml'

    # train.run(data=config, weights=weights, hyp=hyp, batch_size=batch_size, epochs=epochs, imgsz=image_size, project=save_dir)
    train.run(data=config, weights=weights, batch_size=batch_size, epochs=epochs, imgsz=image_size, project=save_dir)
    # python train.py --data Config_P1_orders.yaml --weights yolov5n.pt --batch-size 4 --epochs 3

    # rename outputfile
    # configname = config.split('.yaml')[0]
    # configname = configname.split('Config_')[1]
    # weightsname = weights.split('.pt')[0]
    # # weightsname = 'P1_beta_ID_100_yolov5m6'
    # newfoldername = os.path.join(save_dir, (configname + '_' + str(int(epochs)) + '_' + weightsname))
    # os.rename(os.path.join(save_dir, 'exp'), newfoldername)