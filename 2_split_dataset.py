"""
Created on 18th of Nov. 2021
@author: Robin, Thomas

README:

- This Code will split your dataset into Training- (70% of the dataset), Validation- (20%) and Testing Folder (10%).
  (if you want the numbers to change, head to lines 63-65)
- Define the location of the dataset and train, val and test folder (lines 22-36)
  (train, val and test folder will automatically be created if not existing yet)

Note:
    - requires your dataset to contain a "images"-folder and a "labels"-folder.
"""

import os
import random
import tqdm
import shutil


# Define dataset folder
dataset_PATH = r'C:\MASTERTHESIS\Data\P1_beta_dataset_2021_11_23_annotated_orders'

# make train, val and test into your dataset folder:
train_PATH = r'C:\MASTERTHESIS\Data\P1_beta_dataset_2021_11_23_annotated_orders\train'
if not os.path.exists(train_PATH):
    os.makedirs(train_PATH)

val_PATH = r'C:\MASTERTHESIS\Data\P1_beta_dataset_2021_11_23_annotated_orders\val'
if not os.path.exists(val_PATH):
    os.makedirs(val_PATH)

test_PATH = r'C:\MASTERTHESIS\Data\P1_beta_dataset_2021_11_23_annotated_orders\test'
if not os.path.exists(test_PATH):
    os.makedirs(test_PATH)


"""


"""


# images und labels Folder in test,val und train folder
folder_list = [train_PATH, val_PATH, test_PATH]
for path in folder_list:
    if not os.path.exists(os.path.join(path, 'images')):
        os.makedirs(os.path.join(path, 'images'))
    if not os.path.exists(os.path.join(path, 'labels')):
        os.makedirs(os.path.join(path, 'labels'))

image_files = os.listdir(os.path.join(dataset_PATH, 'images'))
label_files = os.listdir(os.path.join(dataset_PATH, 'labels'))

#zufällige aufteilung der bilder
temp = list(zip(image_files, label_files))
random.shuffle(temp)
image_files, label_files = zip(*temp)

#Aufteilen der Bilder
dataset_length = len(image_files)
train_dataset_length = int(dataset_length * .7)
val_dataset_length = int(dataset_length * .2)
test_dataset_length = int(dataset_length * .1)

#slice training dataset
train_images = image_files[0:train_dataset_length]
train_labels = label_files[0:train_dataset_length]

#slice validation dataset
val_images = image_files[train_dataset_length+1:train_dataset_length + val_dataset_length]
val_labels = label_files[train_dataset_length+1:train_dataset_length + val_dataset_length]

#slice test dataset
test_images = image_files[train_dataset_length + val_dataset_length+1:-1]
test_labels = label_files[train_dataset_length + val_dataset_length+1:-1]

print('move training files')
for image_id, image_file in tqdm.tqdm(enumerate(train_images)):
    label_file = train_labels[image_id]

    image_src = os.path.join(dataset_PATH, 'images', image_file)
    image_dst = os.path.join(train_PATH,  'images', image_file)
    shutil.move(image_src, image_dst)

    label_src = os.path.join(dataset_PATH, 'labels', label_file)
    label_dst = os.path.join(train_PATH, 'labels', label_file)
    shutil.move(label_src, label_dst)

print('move validation files')
for image_id, image_file in tqdm.tqdm(enumerate(val_images)):
    label_file = val_labels[image_id]

    image_src = os.path.join(dataset_PATH, 'images', image_file)
    image_dst = os.path.join(val_PATH,  'images', image_file)
    shutil.move(image_src, image_dst)

    label_src = os.path.join(dataset_PATH, 'labels', label_file)
    label_dst = os.path.join(val_PATH, 'labels', label_file)
    shutil.move(label_src, label_dst)

print('move test files')
for image_id, image_file in tqdm.tqdm(enumerate(test_images)):
    label_file = test_labels[image_id]

    image_src = os.path.join(dataset_PATH, 'images', image_file)
    image_dst = os.path.join(test_PATH, 'images', image_file)
    shutil.move(image_src, image_dst)

    label_src = os.path.join(dataset_PATH, 'labels', label_file)
    label_dst = os.path.join(test_PATH, 'labels', label_file)
    shutil.move(label_src, label_dst)

# Delete empty images and labels folder
delete_images_folder = os.path.join(dataset_PATH, 'images')
shutil.rmtree(delete_images_folder)
delete_labels_folder = os.path.join(dataset_PATH, 'labels')
shutil.rmtree(delete_labels_folder)
