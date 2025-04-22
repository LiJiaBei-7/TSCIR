

path=''

type='dress'
seed=101
lora_rank=128
model=''
dataset_root=''

for epoch in {1..30}
do
    resume_value="${path}/checkpoints/epoch_${epoch}.pt"

    python src/eval_retrieval.py \
        --openai-pretrained \
        --resume $resume_value \
        --eval-mode cirr_test \
        --source-data $type \
        --gpu 3 \
        --stage 2 \
        --lora_rank ${lora_rank} \
        --epoch $epoch \
        --seed $seed \
        --model ${model} \
        --dataset-root {dataset_root} \
        --output-acc-log "${path}/cirr_acc"
done
