import os
import json
import cv2
import pandas as pd
from PIL import Image


def get_img_width_height_pil_from_list(list_img_paths, get_orientation=False):
    """
    Helper function to get image attributes (width, height & orientation) from a list of valid file paths.
    It uses the PIL for reading the image attributes. They might differ from VGG-VIA image attributes.

    :param list_img_paths: A list of valid image file paths.
    :param get_orientation: True or False. Should it return also the integer value from the Exif Orientation field?
    :return: a DataFrame object with 3 or 4 columns.
             The 4th column is the orientation value from the Exif header if you choose get_orientation=True.
             E.g.: 'file_path', 'width_pil', 'height_pil', 'orientation_pil'
    """
    img_statics_dict = dict()

    for i, file_path in enumerate(list_img_paths):
        # If the image can be open, then get width, height, otherwise they both get NaN.
        try:
            img = Image.open(file_path)
            width, height = img.size
        except:
            width = float("nan")
            height = float("nan")
            print('For file', file_path, "couldn't read image width and height! Make sure path is correct or image file is not corrupted")
        # If user wants orientation then get the value by indexing the exif dictionary.
        # This dictionary sadly doesn't come with named keys, but you can directly use the index 274 for orientation.
        # See why index 274 in https://github.com/stark-t/PAI/issues/24
        # Or the official page for exif tags at https://exiv2.org/tags.html, 274 = Exif.Image.Orientation
        # If orientation cannot be read, then assign NaN.
        if get_orientation is True:
            try:
                exif_dict = img._getexif()
                orientation = exif_dict[274]
            except:
                orientation = float("nan")
            # Write into the dictionary
            img_statics_dict[i] = {'file_path': file_path, 'width_pil': width, 'height_pil': height, 'orientation_pil': orientation}
        # If user doesn't want orientation value, then write current image statics into the dictionary.
        else:
            img_statics_dict[i] = {'file_path': file_path, 'width_pil': width, 'height_pil': height}

    # Convert dictionary to DataFrame for final return object.
    df = pd.DataFrame.from_dict(img_statics_dict, orient='index')
    # df = pd.DataFrame(img_statics_dict).transpose() # alternative
    return df


def get_img_width_height_pil_from_dir(dir_img_path,
                                      get_orientation=False,
                                      file_extensions=('.png', 'jpg', '.jpeg', '.JPG')):
    """
    Helper function to get image attributes (width, height & orientation) from a directory containing file images.
    It uses get_img_width_height_pil_from_list() on list of image path from a directory.
    It uses the PIL for reading the image attributes. They might differ from VGG-VIA image attributes.

    :param dir_img_path: path to a folder with images.
    :param get_orientation: True or False. Should it return also the integer value from the Exif Orientation field?
    :param file_extensions: a tuple of file extensions passed to the suffix for .endswith()
    :return: a DataFrame object with 4 or 5 columns.
             The 5th column is the orientation value from the Exif header if you choose get_orientation=True.
             E.g.: 'file_name', 'file_path', 'width_pil', 'height_pil', 'orientation_pil'
    """
    # Read only image files based on given possible file extensions.
    file_names = [file for file in os.listdir(dir_img_path) if file.endswith(file_extensions)]
    # Sort the file names as this assures compatible ordered list on Linux and Windows OS.
    file_names = sorted(file_names)
    file_paths = [os.path.join(dir_img_path, file) for file in file_names]

    df = get_img_width_height_pil_from_list(file_paths, get_orientation)
    df.insert(loc=0, column='file_name', value=file_names)

    return df


def get_img_width_height_cv2_from_list(list_img_paths):
    """
    Helper function to get image width, height using cv2.

    :param list_img_paths: A list of valid image file paths.
    :return: a DataFrame object with 3 columns: 'file_path', 'width_cv2', 'height_cv2'
    """
    img_wh_dict = dict()

    for i, file_path in enumerate(list_img_paths):
        # If the image can be open, then get width, height, otherwise they both get NaN.
        try:
            img = cv2.imread(file_path)
            # ".shape returns a tuple of number of rows, columns"
            # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html#accessing-image-properties
            # ! OpenCV reports first Height then Width, while PIL reports first Width, then Height
            height, width = img.shape[0:2]
        except:
            width = float("nan")
            height = float("nan")
            print('For file', file_path,
                  "couldn't read image width and height! Make sure path is correct or image file is not corrupted")

        img_wh_dict[i] = {'file_path': file_path, 'width_cv2': width, 'height_cv2': height}

    # Convert dictionary to DataFrame for final return object.
    df = pd.DataFrame.from_dict(img_wh_dict, orient='index')
    return df


def get_img_width_height_cv2_from_dir(dir_img_path,
                                      file_extensions=('.png', 'jpg', '.jpeg', '.JPG')):
    """
    Helper function to get image width, height using cv2 from a directory containing file images.

    :param dir_img_path: path to a folder with images.
    :param file_extensions: a tuple of file extensions passed to the suffix for .endswith()
    :return: a DataFrame object with 4 columns: 'file_name', 'file_path', 'width_cv2', 'height_cv2'
    """
    # Read only image files based on given possible file extensions.
    file_names = [file for file in os.listdir(dir_img_path) if file.endswith(file_extensions)]
    # Sort the file names as this assures compatible ordered list on Linux and Windows OS.
    file_names = sorted(file_names)
    file_paths = [os.path.join(dir_img_path, file) for file in file_names]

    df = get_img_width_height_cv2_from_list(file_paths)
    df.insert(loc=0, column='file_name', value=file_names)

    return df


def get_img_width_height_from_via_coco(json_file_path):
    """
    :param json_file_path: path to COCO json file from VIA app - Menu Annotation > Export Annotations (COCO format).
    File format compatible with VGG Image Annotator (VIA) via-2.0.11
    :return: A DataFrame with 3 columns: 'width_via', 'height_via', 'file_name'
    """
    with open(json_file_path) as f:
        json_data = json.load(f)
    # json_data is a dictionary with 5 keys:
    # json_data.keys()
    # dict_keys(['info', 'images', 'annotations', 'licenses', 'categories'])
    # We are interested in the 'images' key because there we find image statics info and file names.
    img_list = json_data['images']
    df = pd.DataFrame.from_dict(img_list)
    # df.columns
    # Return only the columns in which we are interested.
    columns_to_keep = ['file_name', 'width', 'height']
    df = df[columns_to_keep]
    df.rename(columns={'width': 'width_via',
                       'height': 'height_via'},
              inplace=True)
    return df
