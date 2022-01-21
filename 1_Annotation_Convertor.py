"""
Created on 18th of Nov. 2021
@author: Robin, Thomas

README:

- This Code will convert the annotation style of your dataset and put the images and labels into 2 seperate folders.
- Decide the Conversion, define the intention and the Input- and Output-folder (lines 29-46)

Note:
- requires the subfolders of datasetfolder to be the classnames (except UFZ)
- the classnames will then be the annotation of the images as numbers
- Example: First appearing subfolder is named "Diptera". Second one is named "Hymenoptera".
        "Diptera" and "Hymenoptera" are the classnames.
        "Diptera" will be identified by number "0" in the annotation file.
        "Hymenoptera" will be identified by number "1" in the annotation file.
- if you use the conversion for the insect detector, all of the images (with an insect) will have the annotationnumber (0)
- Matching the annotationnumbers with the corresponding classnames happens in step 3: train_PAI in the config file
"""

import os
import tqdm
import xml.etree.ElementTree as ET
import shutil
import json
import numpy as np


#decide which Conversion you want to use (change "False" to "True")
json_to_yolo = False
Pascal_VOC_XML_to_yolo = False
UFZ_to_yolo = True

#define the intention. Do you want to use the Labels for training/testing or for the insect detector? (change "False" to "True")
conversion_training_orders = True
conversion_insect_detector = False

#class balance for UFZ_to_YOLO?
class_balance = False

# define the input folder / dataset path
dataset_PATH = r'C:\MASTERTHESIS\Data\P1_beta_dataset_2021_11_23'
# define the output folder(name): (It will create one if it doesn't exist yet)
output_PATH = r'C:\MASTERTHESIS\Data\P1_beta_dataset_2021_11_23_annotated_orders'
if not os.path.exists(output_PATH):
    os.makedirs(output_PATH)


"""


"""

# create images and labels folder in output folder
folder = [output_PATH]
for path in folder:
    if not os.path.exists(os.path.join(path, 'images')):
        os.makedirs(os.path.join(path, 'images'))
    if not os.path.exists(os.path.join(path, 'labels')):
        os.makedirs(os.path.join(path, 'labels'))

# get all class names
classnames = []
for item in os.listdir(dataset_PATH):
    item_ = os.path.join(dataset_PATH, item)
    if os.path.isfile(item_):
        continue
    else:
        classnames.append(item)
del item, item_


"""
JSON to YOLO!

"""

#json to yolo Conversion:
if json_to_yolo:
    # function to convert kaggle bounding box to yolo bounding box
    def convert(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    # loop through all classes, numerieren der classnames
    for classid, classname in enumerate(classnames):
        # loop through all annotations
        print('processing class:', classid)
        annotation_PATH = os.path.join(dataset_PATH, classname, 'annotations')
        for annotation_file in tqdm.tqdm(os.listdir(annotation_PATH)):
            annotation_data = []
            if annotation_file.endswith('.json'):
                with open(os.path.join(annotation_PATH, annotation_file)) as f:
                    annotation_data = list(json.load(f).items())
                # get image information
                annotation_data_asset = annotation_data[0][-1]
                image_name = annotation_data_asset['name']
                image_PATH = os.path.join(output_PATH, 'images', image_name)

                # copy image to new folder
                if not os.path.exists(image_PATH):
                    src = os.path.join(dataset_PATH, classname, image_name)
                    shutil.copyfile(src, image_PATH)

                # get annotation information
                annotation_data_label = annotation_data[1][-1]
                imagelabel_infos = []
                imagelabel_name = image_name.replace('.jpg', '.txt')
                imagelabel_PATH = os.path.join(output_PATH, 'labels', imagelabel_name)
                # create label file
                with open(imagelabel_PATH, 'w') as f:
                    # loop through every insect in annotation
                    for insect in annotation_data_label:
                        # get label id
                        if conversion_training_orders:
                            label_name = insect['tags'][0]
                            if label_name != classname:
                                for labeldifferid, labeldiffer in enumerate(classnames):
                                    if label_name == labeldiffer:
                                        labelid = labeldifferid
                            else:
                                labelid = classid
                        elif conversion_insect_detector:
                            labelid = 0

                        # get bounding box
                        annotation_box = insect['points']
                        xvalues = []
                        yvalues = []
                        for points in annotation_box:
                            xvalues.append(points['x'])
                            yvalues.append(points['y'])

                        # get image size
                        image_size = annotation_data_asset['size']
                        image_width = image_size['width']
                        image_height = image_size['height']

                        # create yolo bounding box
                        yolo_bounding_box = convert((image_width, image_height),
                                                    (np.min(xvalues), np.max(xvalues),
                                                     np.min(yvalues), np.max(yvalues)))

                        # create info for imagelabel
                        imagelabel_info = str(labelid) + ' ' + \
                                          str(yolo_bounding_box[0]) + ' ' + \
                                          str(yolo_bounding_box[1]) + ' ' + \
                                          str(yolo_bounding_box[2]) + ' ' + \
                                          str(yolo_bounding_box[3]) + '\n'
                        imagelabel_infos.append(imagelabel_info)
                    for insect_label in imagelabel_infos:
                        f.write(insect_label)
                f.close()


"""
Pascal VOC XML to YOLO!

"""

if Pascal_VOC_XML_to_yolo:
    # function to convert pascal.voc bounding box to yolo bounding box
    def convert(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)


    # loop through all classes, numerieren der classnames
    for classid, classname in enumerate(classnames):
        print('processing class:', classid)
        annotation_PATH = os.path.join(dataset_PATH, classname)
        for file_name in tqdm.tqdm(os.listdir(annotation_PATH)):
            annotation_data = []

            if file_name.endswith('.jpg'):
                image_file_PATH = os.path.join(dataset_PATH, classname, file_name)
                annotation_file_PATH = image_file_PATH.replace('.jpg', '.xml')
                if os.path.isfile(annotation_file_PATH):
                    with open(annotation_file_PATH) as f:
                        # open the xml file and get its information
                        tree = ET.parse(annotation_file_PATH)
                        root = tree.getroot()
                        # get the size
                        size = root.find('size')
                        image_width = int(size.find('width').text)
                        image_height = int(size.find('height').text)
                        # get the image_file_name for the order, family, genus, species
                        image_file_name = str(root.find('filename').text)
                        # get the bounding box (xml style)
                        for obj in root.iter('object'):
                            xmlbox = obj.find('bndbox')
                            xmin = int(xmlbox.find('xmin').text)
                            ymin = int(xmlbox.find('ymin').text)
                            xmax = int(xmlbox.find('xmax').text)
                            ymax = int(xmlbox.find('ymax').text)

                        # create yolo bounding box
                        yolo_bounding_box = convert((image_width, image_height), (xmin, xmax, ymin, ymax))

                    f.close()

                    # select class depending on selection above
                    if conversion_training_orders:
                        for i, classname_i in enumerate(classnames):
                            if classname == classname_i:
                                label_name = i
                    elif conversion_insect_detector:
                        label_name = 0

                    # get annotation information
                    image_id = image_file_name.split('\\')[-1]
                    image_id = image_id.split('_')[-1]
                    image_id = image_id.split('.')[0]
                    label_file = os.path.join(path, 'labels', (image_id + '.txt'))
                    image_file = label_file.replace('.txt', '.jpg')
                    image_file = image_file.replace('labels', 'images')
                    with open(label_file, 'w') as ff:
                        # create info for imagelabel
                        imagelabel_info = str(label_name) + ' ' + \
                                          str(yolo_bounding_box[0]) + ' ' + \
                                          str(yolo_bounding_box[1]) + ' ' + \
                                          str(yolo_bounding_box[2]) + ' ' + \
                                          str(yolo_bounding_box[3]) + '\n'
                        annotation_data.append(imagelabel_info)
                        for pollinator_label in annotation_data:
                            ff.write(pollinator_label)
                    ff.close()

                    shutil.copyfile(image_file_PATH, image_file)
                else:
                    # if there is no xml file, select background class
                    label_name = 0
                    shutil.copyfile(image_file_PATH, image_file)



    """
    UFZ Yolo Format to "our" YOLO format!

    """

if UFZ_to_yolo:
    # loop through all classes, numerieren der classnames
    for classid, classname in enumerate(classnames):
        # loop through all images
        i = 0
        print('processing class:', classid)
        image_PATH = os.path.join(dataset_PATH, classname, 'img')
        for image_file in tqdm.tqdm(os.listdir(image_PATH)):
            annotation_data = []
            if conversion_insect_detector:
                classid = 0

            label_name = image_file.split('.')[0]
            label_PATH_src = os.path.join(dataset_PATH, classname, 'annotations', 'yolo_txt', (label_name+'.txt'))
            label_PATH_dst = os.path.join(output_PATH, 'labels', (label_name+'.txt'))

            # read label file and change classes if necessary
            if not os.path.exists(label_PATH_src):
                continue
            with open(label_PATH_src, 'r') as f:
                contents = f.readlines()
            f.close()
            with open(label_PATH_dst, 'w') as f:
                for line in contents:
                    delimiter = ''
                    line_string = delimiter.join(line)
                    line_array = line_string.split()
                    line_array[0] = classid
                    new_line = ' '.join(str(x) for x in line_array)
                    f.write(new_line)
            f.close()

            image_PATH_src = os.path.join(image_PATH, image_file)
            image_PATH_dst = os.path.join(output_PATH, 'images', (label_name+'.jpg'))
            if os.path.exists(label_PATH_src):
                if not class_balance:
                    max_number_of_labels = 1e9
                else:
                    max_number_of_labels = 1000
                i += 1
                if i <= max_number_of_labels:
                    shutil.copyfile(image_PATH_src, image_PATH_dst)
                    # shutil.copyfile(label_PATH_src, label_PATH_dst) #!TODO: MÜSEN WIR UNS ANGUCKEN WENN WIR DIE BALANCED/UNBALANCED SACHE MACHEN WOLLEN
                else:
                    continue