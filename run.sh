#!/bin/bash
export PYTHONUNBUFFERED=1

# Base output directory for experiment artifacts
artifacts_base_dir="./artifacts"

# Generate timestamp-based directory to store results (prevents overwriting)
timestamp=$(date +"%Y%m%d%H%M")

# Experiment configuration parameters
dataset_list=('mnist' 'gtsrb' 'cifar10')
model_list=('lenet' 'resnet18' 'tailnet' 'vgg16')
num_clients=5

echo -e "===== Run Identifier ===== \n ${timestamp}"
for dataset in "${dataset_list[@]}";do
    for model in "${model_list[@]}";do

        artifacts_dir="${artifacts_base_dir}/${timestamp}/${dataset}/${model}"

        python3 Step1_FLtraining.py \
            --num_clients $num_clients \
            --artifacts_dir $artifacts_dir \
            --dataset $dataset \
            --model $model \
            --pretrained \
            --num_rounds 10 

        python3 Step2_gen_watermarks.py \
            --num_clients $num_clients \
            --artifacts_dir $artifacts_dir \
            --dataset $dataset \
            --model $model         

        python3 Step3_traceability.py \
            --num_clients $num_clients \
            --artifacts_dir $artifacts_dir \
            --dataset $dataset \
            --model $model 

    done
done
