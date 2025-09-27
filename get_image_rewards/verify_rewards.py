import typer
from typing import List, Union, Optional
from .gen_reward_value import get_reward_ds_path
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

def verify(
    annotated: str,
):
    rewards = load_from_disk(annotated)
    good_advantages = []
    bad_advantages = []

    for sample in tqdm(rewards, dynamic_ncols=True):
        chosen, rejected = sample['chosen_score'], sample['rejected_score']
        if chosen > rejected:
            good_advantages.append(chosen-rejected)
        else:
            bad_advantages.append(rejected-chosen)

    good_percentage = len(good_advantages) / len(rewards) * 100
    print(f"Total {len(rewards)} samples, good {len(good_advantages)} samples ({good_percentage}%)")
    print(f"Good avg advantage: {sum(good_advantages)/len(good_advantages)}")
    print(f"Bad avg disadvantage: {sum(bad_advantages)/len(bad_advantages)}")

if __name__=="__main__":
    typer.run(verify)
