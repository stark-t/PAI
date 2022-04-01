"""
Created on 25th of March. 2022
@author: Robin
"""

import train
import os

if __name__ == "__main__":
    weights = 'yolov5m6.pt'
    epochs = 10
    image_size = 1280
    batch_size = 4
    save_dir = r"C:\MASTERTHESIS\Results\Evolve"
    config = 'Config_P1_beta_orders.yaml'

    # train.run(data=config, weights=weights, batch_size=batch_size, epochs=epochs, imgsz=image_size, project=save_dir)
    train.run(data=config, weights=weights, imgsz=image_size, batch_size=batch_size, epochs=epochs, project=save_dir, evolve=50)

    # rename outputfile
    configname = config.split('.yaml')[0]
    configname = configname.split('Config_')[1]
    weightsname = weights.split('.pt')[0]
    newfoldername = os.path.join(save_dir, (configname + '_' + str(int(epochs)) + '_' + weightsname))
    os.rename(os.path.join(save_dir, 'exp'), newfoldername)