#!/bin/bash
#SBATCH --job-name=detect_speed
#SBATCH --partition=clara
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:rtx2080ti:1 # 1 rtx2080ti GPU of 11 Gb RAM
#SBATCH --time=02:00:00 
#SBATCH --output=/home/sc.uni-leipzig.de/%u/PAI/detectors/logs_detect_jobs_speed/%j.log
#SBATCH --error=/home/sc.uni-leipzig.de/%u/PAI/detectors/logs_detect_jobs_speed/%j.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT,END

# Start with a clean environment
module purge
# Load the needed modules from the software tree (same ones used when we created the environment)
module load Python/3.9.6-GCCcore-11.2.0
# Activate virtual environment
source ~/venv/yolov5/bin/activate

# Call the helper script session_info.sh which will print in the *.log file info 
# about the used environment and hardware.
source ~/PAI/scripts/cluster/session_info.sh yolov5
# The first and only argument here, passed to $1, is the environment name set at ~/venv/
# Use source instead of bash, so that session_info.sh describes the environment activated in this script 
# (the parent script from which is called). See https://askubuntu.com/a/965496/772524

cd ~/PAI/detectors/yolov5

# Define an array that will contain the elapsed time at each interation
array=()

# Run detection on the test dataset several times in a loop so that we 
# measure detection time and time variability.
# Give the confidence & IoU thresholds that will pass to detect.py 
conf=0.3
iou=0.6

# This path & name will be used for --project and for the txt file with the time records
results_folder=runs/detect/detect_speed_jobs/job_"$SLURM_JOB_ID"_yolov5_small_gpurtx_results_at_"$conf"_iou_"$iou"

# How many times should it run detect.py?
# This will help to get a better time estimate.
# I noticed that the first iteration will be slower (2-3 time slower than the next ones),
# so it will be discarded. In this case, we will average the time of the last 5 iterations.
n_iter=6
for i in $(seq 1 $n_iter)
do
    # Will measure time in seconds %s and nanoseconds %N
    start_time=$(date -u +%s.%N)
    
    python detect.py \
    --weights ~/PAI/detectors/yolov5/runs/train/3219884_yolov5_s_img640_b8_e300_hyp_custom/weights/best.pt \
    --source ~/datasets/P1_Data_sampled/test/images \
    --imgsz 640 \
    --conf-thres $conf \
    --iou-thres $iou \
    --max-det 300 \
    --save-txt \
    --save-conf \
    --nosave \
    --project $results_folder \
    --name iteration_"$i"
    
    end_time=$(date -u +%s.%N)
    
    # Compute the elapsed time with nanosecond resolution per iteration.
    # Perhaps nanosecond level is overkill, but we can do roundings later.
    # https://unix.stackexchange.com/a/314370/313268
    # https://www.xmodulo.com/measure-elapsed-time-bash.html
    elapsed=$(bc <<< $end_time-$start_time)
    
    array+=($elapsed)
done

# Save time array to txt file, values separated by new-line
extension='.txt'
printf "%s\n" ${array[@]} > $results_folder$extension


# Deactivate virtual environment
deactivate

# Print the total elapsed time in the log file
echo '========================================================================'
echo 'Total elapsed time for the entire loop job was:'
sacct -j $SLURM_JOB_ID --format=Elapsed


# Run in terminal with:
# sbatch ~/PAI/scripts/cluster/inference_speed/yolov5_detect_s_640_gpu_rtx_speed_test.sh