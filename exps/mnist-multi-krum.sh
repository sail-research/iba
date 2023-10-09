#!/bin/bash

#$ -l rt_G.small=1 
#$ -l h_rt=8:00:00
#$ -j y
#$ -cwd

# source /etc/profile.d/modules.sh
# module load gcc/9.3.0 python/3.8/3.8.13 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 && python3 -m venv ~/venv/pytorch && source ~/venv/pytorch/bin/activate

python iba_simulated_averaging.py \
--batch-size 256 \
--test-batch-size 256 \
--lr 0.01 \
--gamma 0.998 \
--num_nets 200 \
--fl_round 1500 \
--part_nets_per_round 10 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset mnist \
--model lenet \
--fl_mode fixed-freq \
--attacker_pool_size 0 \
--defense_method multi-krum \
--attack_method blackbox \
--attack_case edge-case \
--attack_freq 10 \
--model_replacement False \
--project_frequency 1 \
--stddev 0.025 \
--eps 2 \
--adv_lr 0.01 \
--prox_attack False \
--poison_type southwest \
--baseline False \
--instance mnist-multi-krum \
--norm_bound 2 \
--atk_model_train_epoch 3 \
--num_dps_attacker 800 \
--attack_alpha 0.5 \
--atk_eps 0.3 \
--atk_test_eps 0.05 \
--attack_portion 1.0 \
--attack_model autoencoder \
--scale 1.0 \
--target_label 0 \
--atk_lr 0.0001 \
--retrain True \
--group LIRA-FL-defenses-blackbox \
--device=cuda