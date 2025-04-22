
path =''

type='dress'
seed=101
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
        --gpu 4 \
        --stage 1\
        --epoch $epoch \
        --seed $seed \
        --model ${model} \
        --dataset-root {dataset_root} \
        --output-acc-log "${path}/cirr_acc"
done
