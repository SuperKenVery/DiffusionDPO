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


def batch_extract(f: Callable[[BatchedSample], Any], batch: Dict[Any, List[Any]]) -> List[Any]:
    keys = list(batch.keys())
    length = len(batch[keys[0]])
    results = []

    for i in range(length):
        sample = { k: batch[k][i] for k in keys }
        results.append(f(sample))
    return results


def annotate(
    dataset: str,
    split: str = "train",
    hps_version: str = "v2.1",
    save_path: Optional[str] = None,
):
    """
    Annotate the reward values with hps model

    :param dataset: The name of the dataset to annotate
    :param split: The dataset split to use (default: "train")
    :param hps_version: The version of HPS model to use (v2.0 or v2.1) (default: "v2.1")
    :param save_path: Path to save the annotated dataset. (default: get_reward_ds_path(dataset))
    """
    print("Loading dataset")
    adapter = DS_ADAPTERS[dataset]
    ds = load_dataset(dataset)[split]

    all_prompts = []
    all_chosen_scores = []
    all_rejected_scores = []
    batch_size = 16
    num_samples = len(ds)

    for start in tqdm(range(0, num_samples, batch_size), dynamic_ncols=True):
        end = min(start + batch_size, num_samples)
        batch = cast(BatchedSample, ds[start:end])

        prompts = batch_extract(adapter.get_prompt, batch)
        chosen_images = batch_extract(adapter.get_chosen_img, batch)
        rejected_images = batch_extract(adapter.get_rejected_img, batch)

        all_images = chosen_images + rejected_images
        all_prompts_for_score = prompts + prompts

        # Here all_images is a list of Image.Image. hpsv2.score is refactored to handle that
        # as well as using DataLoader to speed it up.
        scores = hpsv2.score(all_images, all_prompts_for_score, hps_version=hps_version)
        curr_len = len(prompts)
        chosen_scores = scores[:curr_len]
        rejected_scores = scores[curr_len:]

        all_prompts.extend(prompts)
        all_chosen_scores.extend(chosen_scores)
        all_rejected_scores.extend(rejected_scores)

    print("Saving...")
    reward_ds = Dataset.from_dict({
        "prompt": all_prompts,
        "chosen_score": all_chosen_scores,
        "rejected_score": all_rejected_scores,
    })
    reward_ds.save(save_path or get_reward_ds_path(dataset))

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
