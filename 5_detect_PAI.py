
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import detect
import shutil
import matplotlib
matplotlib.use('TkAgg')

conf_threshold = .50
batch_size = 16
imgsz = 1280
classnames = ['Araneae','Diptera', 'Hemiptera', 'Hymenoptera f.', 'Hymenoptera', 'Lepidoptera', 'Orthoptera']

base_dir = r'C:\MASTERTHESIS\Data\UFZ_field_observation_29_03_22_orders_onlytest\test'
source = base_dir + '\\images'
weights = r"C:\MASTERTHESIS\Results\Training\P1_beta_orders_200_yolov5m6\weights\best.pt"


# make folder to save predictions if not exist
save_dir = base_dir + '\\results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#delete exp folder if exists
if os.path.exists(os.path.join(save_dir, "exp")):
    shutil.rmtree(os.path.join(save_dir, "exp"))

#run yolo to detect images
output = detect.run(weights=weights, source=source, imgsz=(imgsz,imgsz),
                    save_txt=True, nosave=True, conf_thres=conf_threshold, project=save_dir)

# rename labels to predictions
os.rename(os.path.join(save_dir, 'exp', 'labels'), os.path.join(save_dir, 'exp', 'predictions'))
