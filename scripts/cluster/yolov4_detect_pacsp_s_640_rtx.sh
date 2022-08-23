#!/bin/bash
#SBATCH --job-name=detect_yolov4_gpu # give a custom name
#SBATCH --partition=clara-job # use clara-job for GPU usage and clara-cpu for cpu usage
#SBATCH --cpus-per-task=4 # request number of CPUs
#SBATCH --gres=gpu:rtx2080ti:1 # type and number of requested GPUs; Options are rtx2080ti:1 or gpu:v100:1
#SBATCH --mem-per-gpu=11G # RAM per GPU - 11 Gb is for NVIDIA GeForce RTX 2080 Ti; 32 Gb for Nvidia Tesla V100 
#SBATCH --time=01:00:00 # requested time in d-hh:mm:ss e.g. 10-00:00:00 = 10 days, 100:00:00 = 100 hours; 00:30:00 = 30 min
#SBATCH --output=/home/sc.uni-leipzig.de/sv127qyji/PAI/detectors/logs_detect_jobs/%j.log # path for job-id.log file;
#SBATCH --error=/home/sc.uni-leipzig.de/sv127qyji/PAI/detectors/logs_detect_jobs/%j.err # path for job-id.err file;
#SBATCH --mail-type=BEGIN,TIME_LIMIT,END # email options


# Start with a clean environment
module purge
# Load the needed modules from the software tree (same ones used when we created the environment)
module load Python/3.8.6-GCCcore-10.2.0
# Activate virtual environment
source ~/venv/PyTorch_YOLOv4/bin/activate

# Call the helper script session_info.sh which will print in the *.log file info 
# about the used environment and hardware.
source ~/PAI/scripts/cluster/session_info.sh PyTorch_YOLOv4
# The first and only argument here, passed to $1, is the environment name set at ~/venv/
# Use source instead of bash, so that session_info.sh describes the environment activated in this script 
# (the parent script from which is called). See https://askubuntu.com/a/965496/772524

cd ~/PAI/detectors/PyTorch_YOLOv4

# See https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/detect.py
python detect.py \
--weights ~/PAI/detectors/PyTorch_YOLOv4/runs/train/yolov4_pacsp_s_b8_e300_img640_hyp_custom/weights/best_overall.pt \
--source /home/sc.uni-leipzig.de/sv127qyji/datasets/P1_Data_sampled/test/images \
--img-size 640 \
--conf-thres 0.25 \
--iou-thres 0.45 \
--save-txt \
--cfg ~/PAI/detectors/PyTorch_YOLOv4/cfg/yolov4-csp-s-leaky.cfg \
--names ~/PAI/detectors/PyTorch_YOLOv4/data/pai.names \
--output runs/detect/"$SLURM_JOB_ID"_detection_using_3217130_yolov4_pacsp_s_b8_e300_img640_hyp_custom

# Note that, PyTorch_YOLOv4 doesn't have:
# --max-det
# --save-conf
# --nosave
# --name - use --output instead

# Deactivate virtual environment
deactivate

# Run in terminal with:
# sbatch ~/PAI/scripts/cluster/yolov4_detect_pacsp_s_640_rtx.sh