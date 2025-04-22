seed=101
consistency=0.2
adapter_dim=128
loss_ref_txt=1.0
lora_rank=128
warmup=100
topk=10
temperature=0.01
ca_layers=1,3,5,7,9,11 #3,7,11 #2,5,8,11 #1,2,3,4,5,6,7,8,9,10,11 #1,3,5,7,9,11 #1,2,3,4,5,6,7,8,9,10,11
resume='xxx/epoch_x.pt'

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
    --model $model \
    --logs "/logs"