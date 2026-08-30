export LIBERO_HOME=${LIBERO_HOME:-/jfs/jiang/code/unitree/LIBERO}
# export LIBERO_HOME=/path/to/your/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} 
export PYTHONPATH=$(pwd):${PYTHONPATH}


# your_ckpt=/path/to/your/Unifolm-VLA-Libero/checkpoints/pytorch_model.pt
# vlm_pretrained_path=/path/to/your/Unifolm-VLM-Base
your_ckpt=${YOUR_CKPT:-/home/omniverse-2/UnifoLM-VLA-Base/checkpoints/pytorch_model.pt}
vlm_pretrained_path=${VLM_PRETRAINED_PATH:-unitreerobotics/Unifolm-VLM-Base}
ckpt_dir=$(dirname "$your_ckpt")
run_dir=$(dirname "$ckpt_dir")
folder_name=$(basename "$run_dir")
step_name=$(basename "$ckpt_dir")
task_suite_name=${TASK_SUITE_NAME:-libero_spatial}   # libero_goal, libero_object, libero_10, libero_90
num_trials_per_task=${NUM_TRIALS_PER_TASK:-50}
window_size=${WINDOW_SIZE:-2}
unnorm_key="libero_spatial_no_noops"  # libero_goal_no_noops, libero_object_no_noops, libero_10_no_noops, libero_90_no_noops

video_out_path="results/${task_suite_name}/${folder_name}/${step_name}"

DEVICE=${DEVICE:-0}

CUDA_VISIBLE_DEVICES=${DEVICE} python ./experiments/LIBERO/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.vlm-pretrained-path ${vlm_pretrained_path} \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    --args.unnorm-key "$unnorm_key" \
    --args.window-size "$window_size"