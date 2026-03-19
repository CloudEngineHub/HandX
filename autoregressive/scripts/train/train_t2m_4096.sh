#!/bin/bash
#SBATCH --account=bbsg-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g
#SBATCH --job-name=train_t2m

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export WANDB_MODE=disabled
export NCCL_TIMEOUT=1200

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29000 train_t2m_llama.py \
--exp-name coodbook4096_P111M \
--batch-size 128 \
--num-layers 9 \
--embed-dim-gpt 1024 \
--nb-code 4096 \
--n-head-gpt 16 \
--block-size 351 \
--ff-rate 4 \
--drop-out-rate 0.1 \
--resume-pth results/output/FSQ_96len/FSQ_4096_288_patch_haar/net_latest.pth \
--vq-name 4096_288_patch_1gpu \
--out-dir results/output/T2M \
--total-iter 150000 \
--lr-scheduler-type CosineDecayScheduler \
--lr 0.0003 \
--dataname handx \
--down-t 1 \
--depth 3 \
--quantizer FSQ \
--dilation-growth-rate 3 \
--vq-act relu \
--vq-norm LN \
--fps 30 \
--kernel-size 3 \
--use_patcher \
--patch_size 1 \
--patch_method haar \
--text_encode flan-t5-xl \
--pretrained_llama 111M \
--pkeep 1 \
--motion_type vector_272 \
--text_type texts \
--version version1/t2m_60_300 \
--mixed_precision bf16 \
--save-iter-last 1000 \
--gradient_accumulation_steps 1 \
--save-iter 10000 \
--train_split train
