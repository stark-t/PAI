# Image download & annotation processing scripts

Upon request we can provide the data tables containing the URLs for the images used in this project. They were obtained after sampling the public metadata of the GBIF occurrence datasets listed below. Please understand that due to legal and ethical concerns we cannot re-publish the images themselves.

Once you have the tables with URLs, this folder contains the Python scripts required to download the images and generate the YOLO annotation txt files. Start by running the download_img.py script to download the necessary images. Afterward, create the YOLO annotation txt files by executing the make_yolo_txt_files.py script.

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


The images were manually annotated with the VIA Annotation Software:
> Abhishek Dutta and Andrew Zisserman. 2019. The VIA Annotation Software for Images, Audio and Video. In Proceedings of the 27th ACM International Conference on Multimedia (MM ’19), October 21–25, 2019, Nice, France. ACM, New York, NY, USA, 4 pages. https://doi.org/10.1145/3343031.3350535.
