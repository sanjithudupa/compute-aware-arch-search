#!/bin/bash
#SBATCH -N 2                                #number of nodes
#SBATCH -J nvr_elm_llm-ellm:pretrain        #job name
#SBATCH --gpus-per-node 8

if command -v srun &> /dev/null; then
    # on nv cluster
    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    export LOGLEVEL=INFO

    export PATH="$HOME/anaconda3/envs/ellm-clean/bin:$PATH"
    CODEDIR="$HOME/workspace/code/ellm-clean"
    cd $CODEDIR
else
    SLURM_NNODES=1
    head_node_ip=localhost
fi

export OMP_NUM_THREADS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export LM_HARNESS_CACHE_PATH="processed_data/lm_harness_cache"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_ALLOW_CODE_EVAL=1
# Set HuggingFace token to avoid rate limiting (get token from https://huggingface.co/settings/tokens)
# export HF_TOKEN="your_hf_token_here"
# Temporarily disable offline mode to allow dataset download
# export TRANSFORMERS_OFFLINE=1

if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Available GPU count: $gpu_count"

GROUP_NAME=jetlm1.5-2B_stage1
CKPT_PATH="/dataset/videetm/dlm/results/pretrain/jetlm1.5-2B_stage0/trial3/checkpoint/save_19000"
START_MILESTONE=0
END_MILESTONE=30000
INTERVAL=10000
JOB_TYPE=eval
NAME="$JOB_TYPE/$GROUP_NAME"
WORK_DIR="results/$NAME"

read -r -d '' cmd_prefix <<EOF
torchrun --nnodes $SLURM_NNODES --nproc_per_node=$gpu_count --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 \
evaluation.py \
    --seed 42 \
    --config config/ellm/qwen3_1.7B/full_bidirectional_attn.yaml \
    --dist_training.compile False \
    --data.num_workers 4 \
    --data.prefetch_factor 8 \
    --wandb.name ${NAME} \
    --wandb.mode disabled \
    --wandb.commit_strategy "offload:1" \
    --wandb.group ${GROUP_NAME} \
    --wandb.job_type ${JOB_TYPE} \
    --work_dir ${WORK_DIR} \
    --wandb.global_step_shift 20000 \
    --tokenizer.path Qwen/Qwen3-1.7B-Base \
    --dist_training.dp_backend no_dist \
    --offline_eval.ckpt_path ${CKPT_PATH}
EOF

# Running GSM8K with diffusion
cmd_gsm8k="${cmd_prefix} \
    --offline_eval.ckpt_path ${CKPT_PATH}/pytorch_model.bin \
    --eval_batch_size 8 \
    --model.max_seq_len 4096 \
    --offline_eval.evaluator config/eval/gsm8k.yaml \
    --offline_eval.max_new_tokens 2048 \
    --offline_eval.generation_num_chunks 4
"

if command -v srun &> /dev/null; then
    # on nv cluster
    echo $cmd_gsm8k
    srun bash -c "${cmd_gsm8k}"
else
    # on local machine
    echo $cmd_gsm8k
    bash -c "${cmd_gsm8k}"
fi


