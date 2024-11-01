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
cd RobustVLM
python -m vlm_eval.run_evaluation \
--eval_vizwiz \
--verbose \
--dont_save_adv \
--attack pgd0 --eps 4 --steps 100 --mask_out context --mu 1.5 --search_steps 8 \
--vision_encoder_pretrained openai \
--num_samples 800 \
--trial_seeds 42 \
--num_trials 1 \
--shots 4 \
--batch_size 1 \
--results_file  /home/htc/kchitranshi/RobustVLM/res3B \
--model open_flamingo \
--out_base_path /home/htc/kchitranshi/RobustVLM/ \
--vision_encoder_path ViT-L-14 \
--checkpoint_path /home/htc/kchitranshi/SCRATCH/hub/models--openflamingo--OpenFlamingo-4B-vitl-rpj3b-langinstruct/snapshots/11fc728f0c2222355831dfc116c686a852a3fc61/checkpoint.pt \
--lm_path togethercomputer/RedPajama-INCITE-Instruct-3B-v1 \
--lm_tokenizer_path togethercomputer/RedPajama-INCITE-Instruct-3B-v1 \
--precision float32 \
--cross_attn_every_n_layers 2 \
--coco_train_image_dir_path /home/htc/kchitranshi/SCRATCH/COCO/train2014 \
--coco_val_image_dir_path /home/htc/kchitranshi/SCRATCH/COCO/val2014 \
--coco_karpathy_json_path /home/htc/kchitranshi/SCRATCH/COCO/karpathy_coco.json \
--coco_annotations_json_path /home/htc/kchitranshi/RobustVLM/annotations/captions_val2014.json \
--flickr_image_dir_path /home/htc/kchitranshi/SCRATCH/flickr/Images \
--flickr_karpathy_json_path /home/htc/kchitranshi/SCRATCH/flickr/karpathy_flickr30k.json \
--flickr_annotations_json_path /home/htc/kchitranshi/SCRATCH/flickr/dataset_flickr30k_coco_style.json \
--vizwiz_train_image_dir_path /home/htc/kchitranshi/SCRATCH/vizwiz/train \
--vizwiz_test_image_dir_path /home/htc/kchitranshi/SCRATCH/vizwiz/val \
--vizwiz_train_questions_json_path /home/htc/kchitranshi/SCRATCH/vizwiz/train_questions_vqa_format.json \
--vizwiz_train_annotations_json_path /home/htc/kchitranshi/SCRATCH/vizwiz/train_annotations_vqa_format.json \
--vizwiz_test_questions_json_path /home/htc/kchitranshi/SCRATCH/vizwiz/val_questions_vqa_format.json \
--vizwiz_test_annotations_json_path /home/htc/kchitranshi/SCRATCH/vizwiz/val_annotations_vqa_format.json \
--vqav2_train_image_dir_path /home/htc/kchitranshi/SCRATCH/COCO/train2014 \
--vqav2_train_questions_json_path /home/htc/kchitranshi/SCRATCH/vqav2/v2_OpenEnded_mscoco_train2014_questions.json \
--vqav2_train_annotations_json_path /home/htc/kchitranshi/SCRATCH/vqav2/v2_mscoco_train2014_annotations.json \
--vqav2_test_image_dir_path /home/htc/kchitranshi/SCRATCH/COCO/val2014 \
--vqav2_test_questions_json_path /home/htc/kchitranshi/SCRATCH/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
--vqav2_test_annotations_json_path /home/htc/kchitranshi/SCRATCH/vqav2/v2_mscoco_val2014_annotations.json \
--textvqa_image_dir_path /mnt/datasets/textvqa/train_images \
--textvqa_train_questions_json_path /home/htc/kchitranshi/RobustVLM/textvqa/train_questions_vqa_format.json \
--textvqa_train_annotations_json_path /home/htc/kchitranshi/RobustVLM/textvqa/train_annotations_vqa_format.json \
--textvqa_test_questions_json_path /home/htc/kchitranshi/RobustVLM/textvqa/val_questions_vqa_format.json \
--textvqa_test_annotations_json_path /home/htc/kchitranshi/RobustVLM/textvqa/val_annotations_vqa_format.json \
--ok_vqa_train_image_dir_path /home/htc/kchitranshi/SCRATCH/COCO/train2014 \
--ok_vqa_train_questions_json_path /home/htc/kchitranshi/SCRATCH/ok_vqa/OpenEnded_mscoco_train2014_questions.json \
--ok_vqa_train_annotations_json_path /home/htc/kchitranshi/SCRATCH/ok_vqa/mscoco_train2014_annotations.json \
--ok_vqa_test_image_dir_path /home/htc/kchitranshi/SCRATCH/COCO/val2014 \
--ok_vqa_test_questions_json_path /home/htc/kchitranshi/SCRATCH/ok_vqa/OpenEnded_mscoco_val2014_questions.json \
--ok_vqa_test_annotations_json_path /home/htc/kchitranshi/SCRATCH/ok_vqa/mscoco_val2014_annotations.json \
