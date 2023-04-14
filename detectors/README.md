## detectors folder

This folder is the designated space for cloning the git repositories of YOLOv5 and YOLOv7. 
See details and steps at https://github.com/stark-t/PAI/tree/main/scripts/cluster#readme

For each object detector architecture, you will need to follow the installation instructions provided in their respective README files:
- https://github.com/ultralytics/yolov5#readme
- https://github.com/WongKinYiu/yolov7#readme

Before you proceed, make sure you have Git installed on your machine - see this https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

## trained_weights folder

The trained_weights folder contains the output diagnostics and trained `.pt` files for the YOLOv5 and YOLOv7 models. These weights can be used for insect (object) detection tasks on your image dataset.

The folder train_diagnostics_output contains the output diagnostics printed by each detector during training. We also included the `*.err` and `*.log` files as txt files (diagnostic logs) printed by the GPU cluster for each train job at run time.

Once you have the detectors installed, to make use of our weights you can follow these command line suggestions (applicable for Linux):

For YOLOv5 n (nano weights):
```sh
cd detectors/yolov5

python detect.py \
--weights ~/PAI/detectors/weights/yolov5_n_best.pt \
--source ~/path/to/your/folder/with/images \
--img-size 640 \
--conf-thres 0.2 \
--iou-thres 0.5 \
--max-det 300 \
--save-txt \
--save-conf \
--nosave \
--project runs/detect/detect_with_yolov5_n \
--name "results_at_conf_0.2_iou_0.5"
```

For YOLOv5 s (small weights):
```sh
cd detectors/yolov5

python detect.py \
--weights ~/PAI/detectors/weights/yolov5_s_best.pt \
--source ~/path/to/your/folder/with/images \
--img-size 640 \
--conf-thres 0.3 \
--iou-thres 0.6 \
--max-det 300 \
--save-txt \
--save-conf \
--nosave \
--project runs/detect/detect_with_yolov5_s \
--name "results_at_conf_0.3_iou_0.6"
```

For YOLOv7 tiny weights:
```sh
cd detectors/yolov7

python detect.py \
--weights ~/PAI/detectors/weights/yolov7_tiny_best.pt \
--source ~/path/to/your/folder/with/images \
--img-size 640 \
--conf-thres 0.1 \
--iou-thres 0.3 \
--save-txt \
--save-conf \
--nosave \
--project runs/detect/detect_with_yolov7_tiny \
--name "results_at_conf_0.1_iou_0.3"

# Note that, yolov7 doesn't have --max-det argument as yolov5
```