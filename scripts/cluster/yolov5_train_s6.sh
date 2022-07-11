#!/bin/bash
#SBATCH --job-name=train_yolov5 # name for the job;
#SBATCH --partition=clara-job # Request for the Clara cluster;
#SBATCH --nodes=1 # Number of nodes;
#SBATCH --cpus-per-task=32 # Number of CPUs;
#SBATCH --gres=gpu:rtx2080ti:8 # Type and number of GPUs;
#SBATCH --mem-per-gpu=11G # RAM per GPU
#SBATCH --time=50:00:00 # requested time, 50:00:00 = 50 hours;
#SBATCH --output=/home/sc.uni-leipzig.de/sv127qyji/PAI/detectors/logs_train_jobs/%j.log # path for job-id.log file;
#SBATCH --error=/home/sc.uni-leipzig.de/sv127qyji/PAI/detectors/logs_train_jobs/%j.err # path for job-id.err file;
#SBATCH --mail-type=BEGIN,TIME_LIMIT,END # email options;


# Start with a clean environment
module purge

# Activate virtual environment
source ~/venv/yolov5/bin/activate

# Call the helper script session_info.sh which will print in the *.log file info 
# about the used environment and hardware.
bash ~/PAI/scripts/cluster/session_info.sh


module load Python
cd ~/PAI/detectors/yolov5

# Train YOLO by calling train.py
python -m torch.distributed.launch --nproc_per_node 8 train.py \
--weights ~/PAI/detectors/yolov5/weights_v6_1/yolov5s6.pt \
--data ~/PAI/scripts/config_yolov5.yaml \
--hyp hyp.scratch-med.yaml \
--epochs 300 \
--batch-size 64 \
--imgsz 1280 \
--workers 6 \
--name yolov5_s6_b8_e300_hypmed

# Deactivate virtual environment
deactivate