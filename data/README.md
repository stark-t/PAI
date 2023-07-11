# Image download & annotation processing scripts

This folder contains the data tables with image URLs and image annotation and the Python scripts required to download the images and generate their corresponding YOLO annotation txt files. 

Start by running the 'download_img.py' script to download the necessary images from the provided URLs (see 'img_url.txt' and 'syrphid_img_url.txt'). Afterward, create the YOLO annotation txt files by executing the 'make_yolo_txt_files.py' script.

The data tables ('img_url.txt' and 'syrphid_img_url.txt') containing the URLs for the images used in this project were obtained after sampling the public metadata of the GBIF occurrence datasets listed below. Please understand that due to legal and ethical concerns we cannot re-publish the images themselves. However, the provided URLs constitute public information already available on GBIF.

The original occurrence datasets were obtained from the Global Biodiversity Information Facility (GBIF). These tables can be of several Gb each. The following table cites the GBIF occurrence datasets used in this project:

| ID | Insect class           | Citation |
|----|------------------------|----------|
| 0  | Araneae                | GBIF.org (22 October 2021) GBIF Occurrence Download https://doi.org/10.15468/dl.yps72k |
| 1  | Coleoptera             | GBIF.org (01 April 2022) GBIF Occurrence Download  https://doi.org/10.15468/dl.52wu3r |
| 2  | Diptera                | GBIF.org (16 August 2021) GBIF Occurrence Download  https://doi.org/10.15468/dl.aegswt |
| 3  | Hemiptera              | GBIF.org (22 October 2021) GBIF Occurrence Download https://doi.org/10.15468/dl.w6qucx |
| 4  | Hymenoptera            | GBIF.org (16 September 2021) GBIF Occurrence Download https://doi.org/10.15468/dl.pqmjq4 |
| 5  | Hymenoptera_Formicidae | GBIF.org (22 October 2021) GBIF Occurrence Download https://doi.org/10.15468/dl.hvrcmu |
| 6  | Lepidoptera            | GBIF.org (16 September 2021) GBIF Occurrence Download https://doi.org/10.15468/dl.8jm4zg |
| 7  | Orthoptera             | GBIF.org (23 September 2021) GBIF Occurrence Download  https://doi.org/10.15468/dl.pxjjz7 |

A significant proportion of the images, precisely 96.77%, are sourced from the citizen science platforms, iNaturalist and Observation.org, which account for 53.86% and 42.91% of the contributions respectively.

The R scripts used to process the raw GBIF datasets, sample and download the URLs are located at './data/gbif_cleaning/scripts/'. However, running these R scripts is optional as the final sampled images can be downloaded with the 'download_img.py' as mentioned above.

The images were manually annotated with the VIA Annotation Software:
> Abhishek Dutta and Andrew Zisserman. 2019. The VIA Annotation Software for Images, Audio and Video. In Proceedings of the 27th ACM International Conference on Multimedia (MM ’19), October 21–25, 2019, Nice, France. ACM, New York, NY, USA, 4 pages. https://doi.org/10.1145/3343031.3350535.

# Metadata for the files in the data_tables folder

## File img_annotation.txt

This tab-delimited file contains the image and bounding box annotation information for the 8 classes of insects.

### Column Descriptions

- **filename**: Name of the image file
- **used_for_ai**: Indicates if the row (image) should be processed (1=yes) or disregarded (0=no). Some of the downloaded images were not use in the study because they represented insect body parts or not adult stages of the target insect.
- **id_box**: Unique identifier of the bounding box (useful when there are multiple insects per image)
- **label_id**: Numeric identifier of the insect class (from 0 to 7)
- **label_name**: Name of the insect class
    ```
    {'Araneae':'0', 
    'Coleoptera':'1',
    'Diptera':'2',
    'Hemiptera':'3',
    'Hymenoptera':'4',
    'Hymenoptera_Formicidae':'5',
    'Lepidoptera':'6',
    'Orthoptera':'7'}
    ```
- **x_center_rel**: Relative x-coordinate of the center of the bounding box
- **y_center_rel**: Relative y-coordinate of the center of the bounding box
- **width_rel**: Relative width of the bounding box
- **height_rel**: Relative height of the bounding box
- **img_width**: Width of the image in pixels
- **img_height**: Height of the image in pixels
- **folder**: Name of the folder corresponding to the label_id

## File img_url.txt

This tab-delimited file contains the image URLs for the 8 classes of insects.

### Column Descriptions

- **filename**: Name of the image file
- **used_for_ai**: Indicates if the row should be processed (1) or disregarded (0)
- **gbif_id**: Unique identifier for the occurrence in the Global Biodiversity Information Facility (GBIF) datasets
- **media_id**: Unique identifier for the media file associated with the occurrence file in the GBIF datasets
- **url**: URL of the image file as provided by the GBIF datasets
- **license**: License under which the image is available for use as provided by the GBIF datasets
- **label_name**: Name of the insect class
- **folder**: Name of the folder corresponding to the label_name

## Files syrphid_img_annotation.txt and syrphid_img_url.txt

The files syrphid_img_annotation.txt and syrphid_img_url.txt contain image annotation and URL information for the test sample associated with the Syrphidae test dataset. These files share the same column descriptions as mentioned earlier. Note that the `label_name` is always `Diptera` and `label_id` is 1.

