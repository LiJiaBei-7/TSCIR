seed=101
consistency=0.2
adapter_dim=128
loss_ref_txt=1.0
lora_rank=128
warmup=100
topk=10
temperature=0.01
ca_layers=1,3,5,7,9,11 #3,7,11 #2,5,8,11 #1,2,3,4,5,6,7,8,9,10,11 #1,3,5,7,9,11 #1,2,3,4,5,6,7,8,9,10,11
# "/mllm_native/wangyabing.wyb/output/CIR/Adapter/seed_101_cls_consistency_0.2_loss_ref_txt_1.0_strong/lr=0.0001_wd=0.1_agg=True_model=/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-L-14.pt_batchsize=128_workers=8_date=2024-09-04-09-21-43/checkpoints/epoch_8.pt"
# resume='/mllm_native/wangyabing.wyb/output/CIR/Adapter/seed_101_cls_consistency_0.2_loss_ref_txt_1.0_weak_29w/lr=0.0001_wd=0.1_agg=True_model=/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-L-14.pt_batchsize=128_workers=8_date=2024-11-20-03-29-21/checkpoints/epoch_4.pt'
resume='/mllm_native/wangyabing.wyb/output/CIR/Adapter/seed_101_cls_consistency_0.2_loss_ref_txt_1.0_weak_29w_temperature_0.07/lr=0.0001_wd=0.1_agg=True_model=/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-L-14.pt_batchsize=128_workers=8_date=2024-12-24-09-55-38/checkpoints/epoch_2.pt'
# resume='/mllm_native/wangyabing.wyb/output/CIR/Adapter/seed_101_cls_consistency_0.2_loss_ref_txt_1.0_weak_29w_temperature_0.05/lr=0.0001_wd=0.1_agg=True_model=/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-L-14.pt_batchsize=128_workers=8_date=2024-12-29-13-38-10/checkpoints/epoch_6.pt'
# CUDA_VISIBLE_DEVICES=7,8 
python -u src/main.py \
    --save-frequency 1 \
    --stage 2 \
    --train-data="cc/Train_GCC-training_output.csv"  \
    --warmup ${warmup} \
    --batch-size=128 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --ca_layers=$ca_layers \
    --temperature ${temperature} \
    --topk ${topk} \
    --lora_rank=${lora_rank} \
    --openai-pretrained \
    --loss-consistency=${consistency} \
    --seed=${seed} \
    --adapter_dim=${adapter_dim} \
    --loss_ref_txt=${loss_ref_txt} \
    --dist-url="tcp://127.0.0.1:6105" \
    --resume=${resume} \
    --model '/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-L-14.pt' \
    --logs "/mllm_native/wangyabing.wyb/output/CIR/Adapter/ca_${ca_layers}_temp_0.07_epoch_2_seed_${seed}_ft_lora_${lora_rank}_warmup_${warmup}_consistency_${consistency}_gpu8_with_map_hard_mix_${topk}_temperature_${temperature}"