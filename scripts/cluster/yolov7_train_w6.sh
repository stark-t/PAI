#!/bin/bash
#SBATCH --job-name=train_yolov7 # name for the job;
#SBATCH --partition=clara-job # Request for the Clara cluster;
#SBATCH --nodes=1 # Number of nodes;
#SBATCH --cpus-per-task=32 # Number of CPUs;
#SBATCH --gres=gpu:gpu:v100:4 # Type and number of GPUs;
#SBATCH --mem-per-gpu=32G # RAM per GPU;
#SBATCH --time=50:00:00 # requested time, 50:00:00 = 50 hours;
#SBATCH --output=/home/sc.uni-leipzig.de/sv127qyji/PAI/detectors/logs_train_jobs/%j.log # path for job-id.log file;
#SBATCH --error=/home/sc.uni-leipzig.de/sv127qyji/PAI/detectors/logs_train_jobs/%j.err # path for job-id.err file;
#SBATCH --mail-type=BEGIN,TIME_LIMIT,END # email options;


# Start with a clean environment
module purge

# Delete any cache files in the train and val dataset folders that were created from previous jobs.
# This is important when ussing different YOLO versions.
# See https://github.com/WongKinYiu/yolov7/blob/main/README.md#training
rm --force ~/datasets/P1_Data_sampled/train/*.cache
rm --force ~/datasets/P1_Data_sampled/val/*.cache

# Activate virtual environment
source ~/venv/yolov7/bin/activate

# Call the helper script session_info.sh which will print in the *.log file info 
# about the used environment and hardware.
bash ~/PAI/scripts/cluster/session_info.sh yolov7


module load Python
cd ~/PAI/detectors/yolov7

# Train YOLO by calling train.py
python -m torch.distributed.launch --nproc_per_node 8 train.py \
--sync-bn \
--weights ~/PAI/detectors/yolov7/weights_v0_1/yolov7-w6.pt \
--data ~/PAI/scripts/config_yolov5.yaml \
--hyp ~/PAI/scripts/yolo_custom_hyp.yaml \
--epochs 300 \
--batch-size 64 \
--img-size 1280 1280 \
--workers 6 \
--name yolov7_w6_b8_e300_hyp_custom

# Deactivate virtual environment
deactivate