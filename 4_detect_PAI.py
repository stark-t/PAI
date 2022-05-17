
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import detect
import shutil
import matplotlib
matplotlib.use('TkAgg')

conf_threshold = .35
batch_size = 32
imgsz = 1280
classnames = ['Araneae', 'Coleoptera', 'Diptera', 'Hemiptera', 'Hymenoptera', 'Hymenoptera f.', 'Lepidoptera', 'Orthoptera']
# classnames = ['Araneae', 'Diptera', 'Hemiptera', 'Hymenoptera', 'Hymenoptera f.', 'Lepidoptera', 'Orthoptera']
# classnames = ['Insect']

# base_dir = r'C:\MASTERTHESIS\Data\P1_orders\test'
base_dir = r'C:\MASTERTHESIS\Data\Field_observation\test'
# base_dir = r'C:\MASTERTHESIS\Data\Holiday_images\test'
# base_dir = r'C:\MASTERTHESIS\Data\..Challenges\new\try'

source = base_dir + '\\images'
weights = r"C:\MASTERTHESIS\Results\Training\P1_orders_200_yolov5m6_70.817hrs\weights\best.pt"

# make folder to save predictions if not exist
save_dir = base_dir + '\\results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#delete exp folder if exists
if os.path.exists(os.path.join(save_dir, "exp")):
    shutil.rmtree(os.path.join(save_dir, "exp"))

#run yolo to detect images
output = detect.run(weights=weights, source=source, imgsz=(imgsz,imgsz),
                    save_txt=True, nosave=False, conf_thres=conf_threshold, project=save_dir)


# rename labels to predictions
os.rename(os.path.join(save_dir, 'exp', 'labels'), os.path.join(save_dir, 'exp', 'predictions'))
