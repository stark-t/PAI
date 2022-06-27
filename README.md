# PAI: Pollination Artificial Intelligence

## Download object detector

download e.g. yolov5 into PAI/code/yolov5

[ ] add auto downloader according to issue #23

## Prepare training

1. Set-up local variabels in PAI/scripts/utils_config
	- path to original dataset
	- path to sampled dataset (data will be copied in this directory)
	- ...
	
2. Run PAI/scripts/utils_create_dataset
	This will create a test/val balanced and imbalanced training dataset. 
	utils_create_dataset will call utils_datapaths to get all data from the original dataset and utils_datasampling will create a val/test balanced and unbalanced training dataset.