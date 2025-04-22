# path='/mnt_rela/wangyabing.wyb/code/pic2word/logs/lr=0.0001_wd={args.wd}_agg={args.aggregate}_model={args.model}_batchsize={args.batch_size}_workers={args.workers}_date=2024-05-28-08-22-06'
# resume_value="${path}/checkpoints/epoch_7.pt"

# python src/eval_retrieval.py \
#     --openai-pretrained \
#     --resume $resume_value \
#     --eval-mode fashion \
#     --source-data shirt \
#     --gpu 1 \
#     --model '/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-L-14.pt' \
#     --dataset-root '/mnt_rela/wangyabing.wyb/datasets/Fashion-IQ' \
#     --output-acc-log "${path}/acc.txt"

## replace with shirt or dress or toptee



path=''

type='dress'
seed=101

for epoch in {1..30}
do
    resume_value="${path}/checkpoints/epoch_${epoch}.pt"

    python src/eval_retrieval.py \
        --openai-pretrained \
        --resume $resume_value \
        --eval-mode circo_test \
        --source-data $type \
        --gpu 0 \
        --stage 1 \
        --epoch $epoch \
        --seed $seed \
        --model '/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-L-14.pt' \
        --dataset-root '/mnt_rela/wangyabing.wyb/datasets/CRICO' \
        --output-acc-log "${path}/circo_acc.txt"
done
