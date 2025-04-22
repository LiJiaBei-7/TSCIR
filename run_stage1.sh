
seed=101
consistency=0.2
adapter_dim=128
loss_ref_txt=1.0
temperature=0.07 # or 0.05
python -u src/main.py \
    --save-frequency 1 \
    --train-data="cc/Train_GCC-training_output.csv"  \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-4 \
    --stage 1 \
    # --eps=1e-6 \
    # --beta1=0.9 \
    # --beta2=0.98 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --openai-pretrained \
    --loss-consistency=${consistency} \
    --seed=${seed} \
    --temperature ${temperature} \
    --adapter_dim=${adapter_dim} \
    --loss_ref_txt=${loss_ref_txt} \
    --dist-url="tcp://127.0.0.1:3201" \
    --model '/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-L-14.pt' \
    --logs "/mllm_native/wangyabing.wyb/output/CIR/Adapter/CLIP_B_seed_${seed}_cls_consistency_${consistency}_loss_ref_txt_${loss_ref_txt}_weak_29w_temperature_${temperature}"