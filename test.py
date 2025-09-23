import argparse
import io
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

import accelerate
import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel,     StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from utils.iter_dict_batch import iter_dict_batch

DATASET = "ymhao/HPDv2"
MODEL_NAME = "stabilityai/sdxl-turbo"
SPLIT = 'train'

tokenizer = CLIPTokenizer.from_pretrained(
    MODEL_NAME, subfolder="tokenizer"
)

def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples['prompt']:
        if random.random() < 0:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError()
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

def map_train(example, index):
    from get_image_rewards.utils import get_reward_ds_path
    preference_data = load_from_disk(get_reward_ds_path(DATASET))
    train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    sample = example

    if sample['human_preference'][0]==0:
        chosen = sample['image'][1].convert("RGB")
        rejected = sample['image'][0].convert("RGB")
    else:
        chosen = sample['image'][0].convert("RGB")
        rejected = sample['image'][1].convert("RGB")
    chosen, rejected = train_transforms(chosen), train_transforms(rejected)
    combined_pixel_value = torch.cat((chosen, rejected), dim=0)

    example['pixel_values'] = combined_pixel_value
    # example["input_ids"] = tokenize_captions(example)
    example['chosen_reward'] = preference_data['chosen_score'][index]
    example['rejected_reward'] = preference_data['rejected_score'][index]
    assert example['prompt'] == preference_data['prompt'][index]
    return example

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return_d =  {"pixel_values": pixel_values}
    # SDXL takes raw prompts
    return_d["caption"] = [example['prompt'] for example in examples]


    return return_d

def main_test():
    ds = load_dataset(DATASET)[SPLIT]
    dataset = ds.select(range(20_0000, 20_0100)).map(map_train, with_indices=True)
    dataset.set_format(type="torch", columns=["pixel_values"], output_all_columns=True)
    dataloder = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=2,
        num_workers=2,
        drop_last=True
    )

    for batch in tqdm(dataloder):
        print("Going another batch...")

if __name__ == "__main__":
    main_test()
