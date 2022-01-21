"""
Created on 22th of Nov. 2021
@author: Robin, Thomas

README:
- Read and change variables in: Config_ArTaxOr_order.yaml
- In the current file, change variables in lines 14-21
"""

import train
import os

if __name__ == "__main__":
    weights = 'yolov5m6.pt'
    epochs = 50
    batch_size = 4
    save_dir = r"C:\MASTERTHESIS\Results\Training"
    # config = 'Config_ArTaxOr_order.yaml'
    # config = 'Config_ArTaxOr_insect_detector.yaml'
    # config = 'Config_P1_beta_order.yaml'
    config = 'Config_P1_beta_order.yaml'

    # train.run(data=config, weights=weights, batch_size=batch_size, epochs=epochs, project=save_dir)
    train.run(data=config, weights=weights, batch_size=batch_size, epochs=epochs, imgsz=1280)
    # train.run(data=config, weights=weights, batch_size=batch_size, epochs=epochs)

    # rename outputfile
    # configname = config.split('.yaml')[0]
    # configname = configname.split('Config_')[1]
    # weightsname = weights.split('.pt')[0]
    # newfoldername = os.path.join(save_dir, (configname + '_' + str(int(epochs)) + '_' + weightsname))
    # os.rename(os.path.join(save_dir, 'exp'), newfoldername)
