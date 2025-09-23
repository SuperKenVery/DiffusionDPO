import hpsv2
from typing import List, Dict, Literal, Callable, Any, Optional, Tuple, TypeVar
from PIL import Image
from datasets import load_dataset, Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from dataclasses import dataclass


BatchedSample = Dict[Any, List[Any]]
Sample = Dict[Any, Any]
R = TypeVar('R')
Extractor = Callable[[Sample], R]

@dataclass
class DatasetAdapter:
    get_chosen_img: Extractor[Image.Image]
    get_rejected_img: Extractor[Image.Image]
    # transform: Callable[[BatchedSample], BatchedSample] # Remove fields incompatible with DatasetLoader
    get_prompt: Extractor[str]

def get_reward_ds_path(dataset: str) -> str:
    ds_name = dataset.replace("/", "-")
    return f"annotated_rewards/{ds_name}"

def batch_extract(f: Extractor[R], samples: BatchedSample) -> List[R]:
    keys = list(samples.keys())
    length = len(samples[keys[0]])

    result = []
    for i in range(length):
        single_sample = { k: samples[k][i] for k in keys }
        result.append(f(single_sample))
    return result
