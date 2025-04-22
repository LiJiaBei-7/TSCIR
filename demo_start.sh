data_name=toptee ## Choose from coco, imgnet, cirr, dress, shirt, toptee.
path_demo=demo_result/
gpu_id=0
resume='/mllm_native/wangyabing.wyb/output/CIR/Adapter/temp_0.5_epoch_4_seed_101_ft_lora_128_warmup_100_consistency_0.2_gpu8_with_map_hard_mix_10_temperature_0.01/lr=0.0001_wd=0.1_agg=True_model=/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-L-14.pt_batchsize=128_workers=8_date=2025-01-15-09-25-22/checkpoints/epoch_15.pt'
python src/demo.py \
    --openai-pretrained \
    --resume $resume \
    --retrieval-data $data_name \
    --query_file query.json \
    --prompts 'a photo of *' \
    --demo-out $path_demo \
    --gpu $gpu_id \
    --model '/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-L-14.pt'


