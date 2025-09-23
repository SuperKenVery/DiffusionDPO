import hpsv2
from typing import List, Dict, Literal, Callable, Any, Optional, Tuple, TypeVar, TypedDict, cast
from PIL import Image
from datasets import load_dataset, Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from dataclasses import dataclass
import torch
from .utils import DatasetAdapter, BatchedSample, Sample
from torch.utils.data import Dataset, DataLoader

class HPDAdapter(Dataset):
    def __init__(self):
        self.ds = load_dataset("ymhao/HPDv2")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pass


class HpdSample(TypedDict):
    prompt: str
    image_path: str
    raw_annotations: List[Any]
    user_hash: List[Any]
    image: List[Image.Image]
    rank: List[Any]
    human_preference: List[int]

def hpd_get_chosen(_sample: Sample) -> Image.Image:
    sample = cast(HpdSample, _sample)
    if sample['human_preference'][0]==1:
        return sample['image'][0]
    else:
        return sample['image'][1]

def hpd_get_rej(_sample: Sample) -> Image.Image:
    sample = cast(HpdSample, _sample)
    if sample['human_preference'][0]==1:
        return sample['image'][1]
    else:
        return sample['image'][0]


def hpd_get_prompt(sample: Sample) -> str:
    return sample['prompt']


hpd_adapter = DatasetAdapter(
    get_chosen_img=hpd_get_chosen,
    get_rejected_img=hpd_get_rej,
    # transform=hpd_transform,
    get_prompt=hpd_get_prompt
)
