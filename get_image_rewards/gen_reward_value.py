import hpsv2
from typing import List, Dict, Literal, Callable, Any, Optional, Tuple, TypeVar, cast
from PIL import Image
from datasets import load_dataset, Dataset
import typer
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from dataclasses import dataclass

from .hpd import hpd_adapter
from .utils import DatasetAdapter, BatchedSample, get_reward_ds_path

DS_ADAPTERS: Dict[str, DatasetAdapter] = {
    "ymhao/HPDv2": hpd_adapter
}



def annotate2(
    dataset: str,
    split: str = "train",
    hps_version: str = "v2.1",
    save_path: Optional[str] = None,
):
    print("Loading dataset")
    adapter = DS_ADAPTERS[dataset]
    # Only use first 1000 samples
    ds = load_dataset(dataset)[split].select(range(20_0000, 21_0000))

    (prompts, chosen_scores, rejected_scores) = hpsv2.score2(ds, adapter, hps_version="v2.1")

    print("Saving...")
    reward_ds = Dataset.from_dict({
        "prompt": prompts,
        "chosen_score": chosen_scores,
        "rejected_score": rejected_scores,
    })
    reward_ds.save_to_disk(save_path or get_reward_ds_path(dataset))

if __name__=="__main__":
    typer.run(annotate2)
