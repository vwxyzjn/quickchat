#!/bin/bash
#SBATCH --job-name=tgi-swarm
#SBATCH --partition=dev-cluster
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm/logs/%x_%j.out

export volume=/scratch
export model={{model}}
export revision={{revision}}

function unused_port() {
    N=${1:-1}
    comm -23 \
        <(seq "1025" "65535" | sort) \
        <(ss -Htan |
            awk '{print $4}' |
            cut -d':' -f2 |
            sort -u) |
        shuf |
        head -n "$N"
}
if [ -z "$HF_TOKEN" ]; then
  echo "You should provide a Hugging Face token in HF_TOKEN."
  exit 1
fi
export PORT=$(unused_port)
touch {{slurm_hosts_path}}

# for each node
for node in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
    # generate a unique unused port for each node
    PORT=$(unused_port)
    IP_ADDRESS=$(srun --nodes=1 --nodelist=$node hostname -I | awk '{print $1}')
    echo "http://$IP_ADDRESS:$PORT" >> {{slurm_hosts_path}} 
    # run the container
    sudo docker run \
        -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
        --gpus "\"device=$CUDA_VISIBLE_DEVICES"\" --shm-size 1g \
        -v $volume:/data -p $PORT:80 \
        ghcr.io/huggingface/text-generation-inference \
        --model-id $model --revision $revision --max-concurrent-requests 530 --max-total-tokens 8192 --max-input-length 7168 --max-batch-prefill-tokens 7168 &  # run in background
done
wait  # wait for all background jobs to finish