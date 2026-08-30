python deployment/model_server/run_real_eval_server.py \
    --ckpt_path ${CKPT_PATH:-/home/omniverse-2/UnifoLM-VLA-Base/checkpoints/pytorch_model.pt} \
    --port ${PORT:-8777} \
    --unnorm_key ${UNNORM_KEY:-g1_stack_block} \
    --vlm_pretrained_path ${VLM_PRETRAINED_PATH:-unitreerobotics/Unifolm-VLM-Base}