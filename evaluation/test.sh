
CUDA_VISIBLE_DEVICES=6 python test_co3d_1.py \
    --model_path /data/fwl/VGGT_quant/VGGT-1B/model_tracker_fixed_e20.pt \
    --co3d_dir /data1/3d_datasets/ \
    --co3d_anno_dir /data1/fwl/datasets/co3d_v2_annotations/ \
    --dtype quantvggt_w4a4\
    --seed 0 \
    --lac \
    --lwc \
    --exp_name a44_quant \
    --debug_mode all \
    --fast_eval \
    --resume_qs \

