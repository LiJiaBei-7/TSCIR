
path=$checkpoints

# 'dress' 'toptee' 'shirt'
type='shirt'
seed=101
lora_rank=128

for type in 'dress' 
do
    for epoch in {1..30}
    do
        resume_value="${path}/checkpoints/epoch_${epoch}.pt"

        python src/eval_retrieval.py \
            --openai-pretrained \
            --resume $resume_value \
            --eval-mode fashion \
            --source-data $type \
            --gpu 4 \
            --stage 2 \
            --lora_rank ${lora_rank} \
            --seed ${seed} \
            --model '/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-L-14.pt' \
            --dataset-root '/mnt_rela/wangyabing.wyb/datasets/Fashion-IQ' \
            --output-acc-log "${path}/${type}_acc.txt"
    done
done
