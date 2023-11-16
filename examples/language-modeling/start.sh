# python ../gaudi_spawn.py \
#        --world_size 1 run_lora_clm.py \
# q_proj v_proj k_proj o_proj up_proj down_proj gate_proj \
export EXPERIMENTAL_WEIGHT_SHARING=0
python run_lora_clm.py \
       --model_name_or_path meta-llama/Llama-2-7b-hf  \
       --dataset_name timdettmers/openassistant-guanaco \
       --bf16 True \
       --lora_target_modules q_proj v_proj \
       --output_dir ./model_lora_llama \
       --num_train_epochs 2 \
       --per_device_train_batch_size 10 \
       --per_device_eval_batch_size 2 \
       --gradient_accumulation_steps 1 \
       --evaluation_strategy "no"\
       --save_strategy "steps"\
       --save_steps 2000 \
       --save_total_limit 1 \
       --learning_rate 1e-4 \
       --logging_steps 1 \
       --dataset_concatenation \
       --do_train \
       --use_habana \
       --use_lazy_mode \
       --throughput_warmup_steps 3 # --gradient_checkpointing # --profiling_warmup_steps 10 --profiling_steps 1 --profiling_record_shapes False # --gradient_checkpointing