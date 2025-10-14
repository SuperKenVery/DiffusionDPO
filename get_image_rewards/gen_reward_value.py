from typing import List, Dict, Literal, Callable, Any, Optional, Tuple, TypeVar, cast
from PIL import Image
import typer
from tqdm import tqdm
from dataclasses import dataclass
import io
from transformers import AutoProcessor, AutoModel
import rich

device = "cuda"

print("Loading preprocessor")
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "/root/autodl-tmp/test/PickScore_v1"
processor = AutoProcessor.from_pretrained(processor_name_or_path)

def annotate2(
    dataset: str,
    split: str = "train",
    hps_version: str = "v2.1",
    save_path: Optional[str] = None,
):
    import hpsv2
    from .hpd import hpd_adapter
    from .utils import DatasetAdapter, BatchedSample, get_reward_ds_path
    DS_ADAPTERS: Dict[str, DatasetAdapter] = {
        "ymhao/HPDv2": hpd_adapter
    }

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



def get_chosen_and_rejected(sample):
    from torchvision.transforms import functional as vF
    from torchvision import transforms
    import torch

    train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ]
    )

    jpg_0s = [train_transforms(Image.open(io.BytesIO(img_data))) for img_data in sample['jpg_0']]
    jpg_1s = [train_transforms(Image.open(io.BytesIO(img_data))) for img_data in sample['jpg_1']]

    chosen = []
    rejected = []
    for idx, label in enumerate(sample['label_0']):
        if label==1:
            chosen.append(jpg_0s[idx])
            rejected.append(jpg_1s[idx])
        else:
            chosen.append(jpg_1s[idx])
            rejected.append(jpg_0s[idx])

    processed_prompt = processor(text=sample['caption'],padding=True, truncation=True, max_length=77, return_tensors="pt",)
    processed_chosen = processor(images=chosen,padding=True, truncation=True, max_length=77, return_tensors="pt",do_rescale=False,)
    processed_rejected = processor(images=rejected,padding=True, truncation=True, max_length=77, return_tensors="pt",do_rescale=False,)
    return {
        'caption': sample['caption'],
        'caption_input_ids': processed_prompt['input_ids'],
        'caption_attention_mask': processed_prompt['attention_mask'],
        'chosen_pixel_values': processed_chosen['pixel_values'],
        'rejected_pixel_values': processed_rejected['pixel_values'],
    }

def annotate_with_pickscore(
    dataset: str,
    split: str = "train",
    save_path: Optional[str] = None,
    batch_size: int = 12,
    num_workers: int = 16,
):
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torch
    # torch.multiprocessing.set_start_method('spawn')
    from datasets import load_dataset, Dataset
    from .utils import get_reward_ds_path


    all_prompts = []
    all_chosen_scores = []
    all_rejected_scores = []


    print("Loading pickscore")
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
    before = model.logit_scale.exp()

    print("Loading dataset")
    ds = load_dataset(dataset)[split].with_transform(get_chosen_and_rejected) #.select(range(10000, 10000+1000))
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


    for batch in tqdm(loader):
        chosen_image_inputs = {'pixel_values': batch['chosen_pixel_values'].to(device)}
        rejected_image_inputs = {'pixel_values': batch['rejected_pixel_values'].to(device)}
        text_inputs = {
            'input_ids': batch['caption_input_ids'].to(device),
            'attention_mask': batch['caption_attention_mask'].to(device),
        }

        chosen_embs = model.get_image_features(**chosen_image_inputs)
        chosen_embs = chosen_embs / torch.norm(chosen_embs, dim=-1, keepdim=True)

        rejected_embs = model.get_image_features(**rejected_image_inputs)
        rejected_embs = rejected_embs / torch.norm(rejected_embs, dim=-1, keepdim=True)

        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        assert chosen_embs.shape == rejected_embs.shape == text_embs.shape == (batch_size, 1024)
        chosen_scores = before * torch.sum(text_embs * chosen_embs, dim=-1)
        rejected_scores = before * torch.sum(text_embs * rejected_embs, dim=-1)

        all_prompts += batch['caption']
        all_chosen_scores += chosen_scores.tolist()
        all_rejected_scores += rejected_scores.tolist()

    print("Saving...")
    reward_ds = Dataset.from_dict({
        "prompt": all_prompts,
        "chosen_score": all_chosen_scores,
        "rejected_score": all_rejected_scores,
    })
    reward_ds.save_to_disk(save_path or get_reward_ds_path(dataset))

def remove_ties_from_pickscore_annotated(
    dataset: str,
    split: str = "train",
    preference_path: Optional[str] = None,
    save_path: Optional[str] = None
):
    from datasets import load_from_disk, load_dataset, Dataset
    from .utils import get_reward_ds_path

    ds = load_from_disk(preference_path or get_reward_ds_path(dataset))
    ori_ds = load_dataset(dataset)[split]
    assert len(ds)==len(ori_ds), "Length not equal, maybe already filtered"

    all_prompts = []
    all_chosen_scores = []
    all_rejected_scores = []

    for idx, (data, pref) in enumerate(zip(tqdm(ori_ds), ds)):
        if data['label_0'] != 0.5:
            all_prompts.append(pref['prompt'])
            all_chosen_scores.append(pref['chosen_score'])
            all_rejected_scores.append(pref['rejected_score'])

    print("Saving to disk...")
    filtered_reward_ds = Dataset.from_dict({
        "prompt": all_prompts,
        "chosen_score": all_chosen_scores,
        "rejected_score": all_rejected_scores,
    })
    filtered_reward_ds.save_to_disk(save_path or preference_path or get_reward_ds_path(dataset))

if __name__=="__main__":
    # typer.run(annotate2)
    # typer.run(annotate_with_pickscore)
    typer.run(remove_ties_from_pickscore_annotated)
