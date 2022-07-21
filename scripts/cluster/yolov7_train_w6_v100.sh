#!/bin/bash
#SBATCH --job-name=train_yolov7 # name for the job;
#SBATCH --partition=clara-job # Request for the Clara cluster;
#SBATCH --nodes=1 # Number of nodes;
#SBATCH --cpus-per-task=32 # Number of CPUs;
#SBATCH --gres=gpu:v100:4 # Type and number of GPUs;
#SBATCH --mem-per-gpu=32G # RAM per GPU;
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
module load Python/3.9.6-GCCcore-11.2.0
# Activate virtual environment
source ~/venv/yolov7/bin/activate

# Call the helper script session_info.sh which will print in the *.log file info 
# about the used environment and hardware.
source ~/PAI/scripts/cluster/session_info.sh yolov7
# The first and only argument here, passed to $1, is the environment name set at ~/venv/
# Use source instead of bash, so that session_info.sh describes the environment activated in this script 
# (the parent script from which is called). See https://askubuntu.com/a/965496/772524


# Train YOLO by calling train.py
cd ~/PAI/detectors/yolov7
python -m torch.distributed.launch --nproc_per_node 4 train.py \
--sync-bn \
--weights ~/PAI/detectors/yolov7/weights_v0_1/yolov7-w6.pt \
--data ~/PAI/scripts/config_yolov5.yaml \
--hyp ~/PAI/scripts/yolo_custom_hyp.yaml \
--epochs 300 \
--batch-size 64 \
--img-size 640 640 \
--workers 6 \
--nosave \
--name yolov7_w6_b8_e300_img640_hyp_custom


# Deactivate virtual environment
deactivate