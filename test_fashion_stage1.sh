
path=''

# 'dress' 'toptee' 'shirt'
type='shirt'
seed=101
model=''
dataset_root=''

for type in 'dress' 'toptee' 'shirt'
do
    for epoch in {1..30}
    do
        resume_value="${path}/checkpoints/epoch_${epoch}.pt"

        python src/eval_retrieval.py \
            --openai-pretrained \
            --resume $resume_value \
            --eval-mode fashion \
            --source-data $type \
            --gpu 3 \
            --stage 1 \
            --seed ${seed} \
            --model ${model} \
            --dataset-root {dataset_root} \
            --output-acc-log "${path}/${type}_acc.txt"
    done
done
