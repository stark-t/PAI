#!/bin/bash
#SBATCH --job-name=train_yolov4_scaled # name for the job;
#SBATCH --partition=clara-job # Request for the Clara cluster;
#SBATCH --nodes=1 # Number of nodes;
#SBATCH --cpus-per-task=32 # Number of CPUs;
#SBATCH --gres=gpu:rtx2080ti:8 # Type and number of GPUs;
#SBATCH --mem-per-gpu=11G # RAM per GPU;
#SBATCH --time=50:00:00 # requested time, 50:00:00 = 50 hours;
#SBATCH --output=/home/sc.uni-leipzig.de/sv127qyji/PAI/detectors/logs_train_jobs/%j.log # path for job-id.log file;
#SBATCH --error=/home/sc.uni-leipzig.de/sv127qyji/PAI/detectors/logs_train_jobs/%j.err # path for job-id.err file;
#SBATCH --mail-type=BEGIN,TIME_LIMIT,END # email options;


# Delete any cache files in the train and val dataset folders that were created from previous jobs.
# This is important when ussing different YOLO versions.
# See https://github.com/WongKinYiu/yolov7/blob/main/README.md#training
rm --force ~/datasets/P1_Data_sampled/train/*.cache
rm --force ~/datasets/P1_Data_sampled/val/*.cache

module purge
module load PyTorch/1.7.1-fosscuda-2019b-Python-3.7.4
module load TensorFlow/2.4.0-fosscuda-2019b-Python-3.7.4
module load OpenCV/4.2.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
module load torchvision/0.8.2-fosscuda-2019b-Python-3.7.4-PyTorch-1.7.1
module load tqdm

# Activate virtual environment
source ~/venv/ScaledYOLOv4/bin/activate

cd ~/PAI/detectors/ScaledYOLOv4

# Train YOLO by calling train.py
python -m torch.distributed.launch --nproc_per_node 8 train.py \
--weights ~/PAI/detectors/ScaledYOLOv4/weights/yolov4-p6.pt \
--data ~/PAI/scripts/config_yolov5.yaml \
--hyp ~/PAI/scripts/yolo_custom_hyp.yaml \
--epochs 300 \
--batch-size 32 \
--img-size 1280 1280 \
--nosave \
--name yolov4_scaled_p6_b4_e300_hyp_custom

# Deactivate virtual environment
deactivate