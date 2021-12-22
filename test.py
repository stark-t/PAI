# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:35:57 2021

@author: Rob
"""

#README: change: lines 13-21

import os
import detect

n_classes = 1   
conf_threshold = .75
batch_size = 4
imgsz = 640
datasetname = "Urlaubsbilder"
#source = r"C:\Users\Rob\Desktop\PRAKTIKUM\2021-08-04\Centaurea-jacea-bs-01"
#source = r"N:\PROJECTS\YOLOv5_TRIAL\ArTaxOr\test1\images"
source = r"N:\PROJECTS\YOLOv5_TRIAL\ArTaxOr_YOLO_Trial\test\images"
save_dir = r"N:\PROJECTS\YOLOv5_TRIAL\yolov5\runs\detect\test"

if n_classes == 1:
    weights = r"N:\PROJECTS\YOLOv5_TRIAL\yolov5\runs\train\weights_100epochs\1cl_100epochs_best.pt"
elif n_classes == 7:
    weights = r"N:\PROJECTS\YOLOv5_TRIAL\yolov5\runs\train\weights_100epochs\7cl_100epochs_best.pt"
else:
    print("error: select model")
modelname = weights.split('\\')[-1]
modelname = modelname.split('_best.pt')[0]

#make folder to save predictions if not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#define label path and detected images path
label_dir = os.path.join(save_dir, 'exp', 'labels')
detected_insects_dir = label_dir.replace('labels', 'images')
    
#run yolo to detect images from helmholtz
output = detect.run(weights=weights, source=source, imgsz=imgsz, save_txt=True, conf_thres=conf_threshold, project=save_dir)

#rename outputfile
newfoldername = os.path.join(save_dir, (datasetname+'_'+modelname+'_'+str(int(conf_threshold*100))+'%'))
os.rename(os.path.join(save_dir, 'exp'), newfoldername)
