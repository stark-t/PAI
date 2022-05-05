import os
import glob
import numpy as np
import tqdm

image_dirs = glob.glob(r'F:\202105_PAI\data\field_observation\images\*')
label_dirs = glob.glob(r'F:\202105_PAI\data\field_observation\labels\*')

print('\nThere are {} images and {} labels'.format(len(image_dirs), len(label_dirs)))

n_BB = 1
classes = []

for file_number, file_name in enumerate(label_dirs):
    #read and get label and prediction
    labels = []
    if os.path.exists(file_name):
        with open(file_name, 'r') as lf:
            label_lines_str = lf.readlines()
            for i, info in enumerate(label_lines_str):
                label_info_str = label_lines_str[i].split(" ")
                label_floats = [float(f) for f in label_info_str]
                labels.append(label_floats)

                n_BB += 1
                class_ = label_floats[0]
                classes.append(class_)

print('In {} labels are {} insects with BBs'.format(len(label_dirs), n_BB))
unique_classes, unique_counts = np.unique(classes, return_counts=True)
for class_ in unique_classes:
    print('For class {} there are {} BBs'.format(int(class_), unique_counts[int(class_)]))




