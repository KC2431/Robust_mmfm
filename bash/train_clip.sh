#!/bin/bash
#SBATCH --job-name=Search
#SBATCH --chdir=/home/htc/kchitranshi/      # Navigate to the working directory where your script lies
#SBATCH --output=/home/htc/kchitranshi/SCRATCH/%j.log     # Standard output and error log
#
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=gpu  # Specify the desired partition, e.g. gpu or big
#SBATCH --exclude=htc-gpu[020-023,037,038] # Only A40 GPU
#SBATCH --time=0-20:00:00 # Specify a Time limit in the format days-hrs:min:sec. Use sinfo to see node time limits
#SBATCH --ntasks=1
#
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chitranshi@zib.de

echo 'Getting node information'
date;hostname;id;pwd

echo 'Setting LANG to en_US.UTF-8'
LANG=en_US.UTF-8

echo 'Activating virtual environment'
source MyEnv/bin/activate
which python
java -version

echo 'Enabling Internet Access'
export https_proxy=http://squid.zib.de:3128
export http_proxy=http://squid.zib.de:3128

echo 'Print GPUs'
/usr/bin/nvidia-smi

echo 'Running script'
cd Robust_mmfm
python vlm_eval/clip_train.py --num_epochs 10 --data_seeds 107 108 --data_name medium --save_model --save_model_path /home/htc/kchitranshi/SCRATCH/trained_clip_models/APGD_4/
