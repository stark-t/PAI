#!/bin/bash
#SBATCH --job-name=detect_yolov4_gpu # give a custom name
#SBATCH --partition=clara-job # use clara-job for GPU usage and clara-cpu for cpu usage
#SBATCH --cpus-per-task=4 # request number of CPUs
#SBATCH --gres=gpu:rtx2080ti:1 # type and number of requested GPUs; Options are rtx2080ti:1 or gpu:v100:1
#SBATCH --mem-per-gpu=11G # RAM per GPU - 11 Gb is for NVIDIA GeForce RTX 2080 Ti; 32 Gb for Nvidia Tesla V100 
#SBATCH --time=20:00:00 # requested time in d-hh:mm:ss e.g. 10-00:00:00 = 10 days, 100:00:00 = 100 hours; 00:30:00 = 30 min
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

# This will create a folder in  ~/PAI/detectors/PyTorch_YOLOv4/runs/detect with the name given in the --output argument,
# job_"$SLURM_JOB_ID"_loop_detect_on_3217130_yolov4_pacsp_s_b8_e300_img640_hyp_custom
# In that folder will create several other folders containing the label & images files.
# The names of these folders correspond to different conf and IoU levels.

for conf in $(seq 0.1 0.1 0.9)
do
    for iou in $(seq 0.1 0.1 0.9)
    do
        # See https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/detect.py
        python detect.py \
        --weights ~/PAI/detectors/PyTorch_YOLOv4/runs/train/yolov4_pacsp_s_b8_e300_img640_hyp_custom/weights/best.pt \
        --source /home/sc.uni-leipzig.de/sv127qyji/datasets/P1_Data_sampled/test/images \
        --img-size 640 \
        --conf-thres "$conf" \
        --iou-thres "$iou" \
        --save-txt \
        --cfg ~/PAI/detectors/PyTorch_YOLOv4/cfg/yolov4-csp-s-leaky.cfg \
        --names ~/PAI/detectors/PyTorch_YOLOv4/data/pai.names \
        --output runs/detect/job_"$SLURM_JOB_ID"_loop_detect_on_3217130_yolov4_pacsp_s_b8_e300_img640_hyp_custom/results_at_conf_"$conf"_iou_"$iou"
    done
done
# Note that, PyTorch_YOLOv4 doesn't have:
# --max-det
# --save-conf
# --nosave
# --name - use --output instead
# --project, use --output instead
# Similar to yolov7, it also doesn't have --max-det argument as yolov5

# For --output, need to create a parent folder, similar to what the --project argument from YOLOv5 & 7 does.

# Deactivate virtual environment
deactivate

# Run in terminal with:
# sbatch ~/PAI/scripts/cluster/yolov4_detect_pacsp_s_640_rtx.sh