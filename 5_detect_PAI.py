"""
Created on 1st of Dec. 2021
@author: Robin, Thomas

README:

- change variables in lines 15-28
"""

import os
import detect
import shutil
import tqdm

#decide if you want bounding boxes in the detected images
img_bounding_boxes = True
#decide if you want to delete the predictions (all images from source with predictions) (for checkup for example)
delete_prediction_images = False

conf_threshold = .50
batch_size = 16
imgsz = 1280
source = r"C:\MASTERTHESIS\Data\Testdatensatz_Robin"
# save_dir = r"C:\MASTERTHESIS\Results\insect_detector"
# save_dir = r"C:\MASTERTHESIS\Results\pollinator_detector"
save_dir = r"C:\MASTERTHESIS\Results\order_classification"
weights = r"C:\MASTERTHESIS\Results\Training\P1_beta_orders_200_yolov5m6\weights\best.pt"

"""


"""

#make folder to save predictions if not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#define label path and detected images path
label_dir = os.path.join(save_dir, 'exp', 'labels')
detected_insects_dir = label_dir.replace('labels', 'images')

#run yolo to detect images
output = detect.run(weights=weights, source=source, imgsz=(imgsz, imgsz), save_txt=True, conf_thres=conf_threshold, project=save_dir)

#make path if not exist
if not os.path.exists(detected_insects_dir):
    os.makedirs(detected_insects_dir)

#loop through all images with labels to copy fitting images
for label_file_name in tqdm.tqdm(os.listdir(label_dir)):
    #get label file
    label_file = os.path.join(label_dir, label_file_name)
    #get source image file
    if img_bounding_boxes == False:
        old_img_file = os.path.join(source, label_file_name)
        old_img_file = old_img_file.replace('.txt', '.jpg')
    else:
        old_img_file = os.path.join(save_dir, 'exp', label_file_name)
        old_img_file = old_img_file.replace('.txt', '.jpg')
    #get new image file
    detected_insects_file = os.path.join(detected_insects_dir, label_file_name)
    detected_insects_file = detected_insects_file.replace('.txt', '.jpg')
    #copy images to new directory
    shutil.copyfile(old_img_file, detected_insects_file)

#delete redundant images in exp
if delete_prediction_images:
    for image_to_delete in tqdm.tqdm(os.listdir(os.path.join(save_dir, 'exp'))):
        if image_to_delete.endswith('.jpg'):
            delete_redundant_image = os.path.join(save_dir, 'exp', image_to_delete)
            delete_redundant_image = delete_redundant_image.replace('.txt', '.jpg')
            os.remove(delete_redundant_image)
        elif image_to_delete.endswith('.JPG'):
            delete_redundant_image = os.path.join(save_dir, 'exp', image_to_delete)
            delete_redundant_image = delete_redundant_image.replace('.txt', '.JPG')
            os.remove(delete_redundant_image)

#rename outputfile
sourcename = source.split("\\")[3]
weightsname = weights.split("\\")[4]
newfoldername = os.path.join(save_dir, (sourcename + "_" + weightsname + "_best_" + 'Threshhold_' + str(int(conf_threshold*100)) + '%'))
os.rename(os.path.join(save_dir, 'exp'), newfoldername)
