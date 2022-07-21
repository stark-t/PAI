#!/bin/bash
#SBATCH --job-name=train_yolov4 # name for the job;
#SBATCH --partition=clara-job # Request for the Clara cluster;
#SBATCH --nodes=1 # Number of nodes;
#SBATCH --cpus-per-task=32 # Number of CPUs;
#SBATCH --gres=gpu:rtx2080ti:8 # Type and number of GPUs;
#SBATCH --mem-per-gpu=11G # RAM per GPU;
#SBATCH --time=5-00:00:00 # requested time in d-hh:mm:ss
#SBATCH --output=/home/sc.uni-leipzig.de/sv127qyji/PAI/detectors/logs_train_jobs/%j.log # path for job-id.log file;
#SBATCH --error=/home/sc.uni-leipzig.de/sv127qyji/PAI/detectors/logs_train_jobs/%j.err # path for job-id.err file;
#SBATCH --mail-type=BEGIN,TIME_LIMIT,END # email options;


# Delete any cache files in the train and val dataset folders that were created from previous jobs.
# This is important when ussing different YOLO versions.
# See https://github.com/WongKinYiu/yolov7/blob/main/README.md#training
rm --force ~/datasets/P1_Data_sampled/train/*.cache
rm --force ~/datasets/P1_Data_sampled/val/*.cache


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


# Train YOLO by calling train.py
cd ~/PAI/detectors/PyTorch_YOLOv4
python -m torch.distributed.launch --nproc_per_node 8 train.py \
--sync-bn \
--cfg ~/PAI/detectors/PyTorch_YOLOv4/cfg/yolov4-csp-s-leaky.cfg \
--weights ~/PAI/detectors/PyTorch_YOLOv4/weights/yolov4-csp-s-leaky.weights \
--data ~/PAI/scripts/config_yolov5.yaml \
--hyp ~/PAI/scripts/yolo_custom_hyp.yaml \
--epochs 300 \
--batch-size 64 \
--img-size 640 640 \
--workers 3 \
--name yolov4_pacsp_s_b8_e300_img640_hyp_custom


# Deactivate virtual environment
deactivate