

path=''

type='dress'
seed=101
lora_rank=128

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
        --model '/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-L-14.pt' \
        --dataset-root '/mnt_rela/wangyabing.wyb/datasets/CIRR/CIRR-cirr_dataset' \
        --output-acc-log "${path}/cirr_acc"
done
