

path=''
model=''
dataset_root=''
type='dress'
seed=101
lora_rank=128

for epoch in {1..30}
do
    resume_value="${path}/checkpoints/epoch_${epoch}.pt"

    python src/eval_retrieval.py \
        --openai-pretrained \
        --resume $resume_value \
        --eval-mode circo_test \
        --source-data $type \
        --gpu 0 \
        --stage 2 \
        --epoch $epoch \
        --lora_rank ${lora_rank} \
        --seed $seed \
        --model ${model} \
        --dataset-root ${dataset_root} \
        --output-acc-log "${path}/circo_acc.txt"
done
