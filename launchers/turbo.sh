export MODEL_NAME="stabilityai/sdxl-turbo"
export VAE="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="ymhao/HPDv2"


# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~30 hours / 200 steps

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=12 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=57 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 5 \
  --beta_dpo 5000 \
   --sdxl --resolution 512 --proportion_empty_prompts 0 \
  --output_dir="trainings/reproduce-2" \
  --caption_column=prompt \
  --image_column=image \
  --ds_start_idx=200000 \
  --ds_end_idx=210000
