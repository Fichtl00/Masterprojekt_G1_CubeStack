#!/bin/bash
# UnifoLM-VLA finetuning "von Scratch" (VLM-Backbone, kein G1-VLA-Checkpoint) auf
# unitreerobotics/G1_Dex3_BlockStacking_Dataset. Siehe
# Anleitung/Masterprojekt/unifolm_vla_scratch_finetuning_g1_dex3_blockstacking.md

# model
Framework_name=unifolm_vla
base_vlm=/home/omniverse-2/UnifoLM-VLM-Base
model_type=qwen2_5_vl
freeze_module_list='qwen_vl_interface'
window_size=1

# dataset (RLDS root -- must contain g1_dex3_blockstacking/1.0.0/)
oxe_data_root=/home/omniverse-2/IsaacLab/Datasets/g1_dex3_blockstacking_rlds
data_mix=g1_dex3_blockstacking

# run save path
run_root_dir=/home/omniverse-2/Groot/outputs_unifolm_vla
run_id=g1_dex3_blockstacking_scratch

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file src/unifolm_vla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  --mixed_precision bf16 \
  src/unifolm_vla/training/train_unifolm_vla.py \
  --config_yaml ./src/unifolm_vla/config/training/unifolm_vla_train.yaml \
  --framework.framework_py ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.model_type ${model_type} \
  --framework.action_model.action_dim 28 \
  --framework.action_model.state_dim 28 \
  --framework.action_model.action_horizon 16 \
  --framework.action_model.future_action_window_size 15 \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.window_size ${window_size} \
  --datasets.vla_data.per_device_batch_size 2 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 30000 \
  --trainer.shuffle_buffer_size 500 \
  --trainer.save_interval 2000 \
  --trainer.use_wrist_image True \
  --trainer.use_proprio True \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 500 \
  --trainer.learning_rate.base 4e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project unifolm_vla_g1_dex3 \
  --wandb_entity local
