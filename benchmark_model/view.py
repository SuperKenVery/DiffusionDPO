import typer
from diffusers import AutoPipelineForText2Image
from datasets import load_dataset
from tqdm import tqdm
import hpsv2
from typing import List, Optional, Union, Tuple, Callable

import os
import torch

app = typer.Typer()

@app.command()
def view(
    models: List[str],
    dataset: str = "ymhao/HPDv2",
    split: str = "test",
    num_samples: int = 10,
):
    ds = load_dataset(dataset)[split].select(range(num_samples))
    prompts = ds["prompt"]
    os.makedirs("view", exist_ok=True)
    for model in tqdm(models):
        pipe = AutoPipelineForText2Image.from_pretrained(model).to("cuda")
        images = pipe(
            prompts,
            num_inference_steps=4,
        ).images
        for i, image in enumerate(images):
            model_name = model.replace("/", "-")
            image.save(f"view/{model_name}-{i}.png")

if __name__ == "__main__":
    app()
