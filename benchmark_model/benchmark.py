import typer
from diffusers import DiffusionPipeline
from datasets import load_dataset
from tqdm import trange
import hpsv2
from typing import List, Optional, Union, Tuple, Callable

def extract_prompt(examples):
    return {"prompt": examples['prompt']}

def benchmark_one_model(
    model_path: str,
    dataset,
    batch_size: int = 8,
):
    model = DiffusionPipeline.from_pretrained(model_path).to("cuda")
    model.set_progress_bar_config(disable=True)

    prompts = [sample['prompt'] for sample in dataset]

    scores = []
    for i in trange(0, len(prompts), batch_size, desc=f"Evaluating {model_path}"):
        chunk_prompts = prompts[i:i + batch_size]
        chunk_images = model(chunk_prompts, output_type="pil").images  # This returns a list of PIL.Image.Image objects

        chunk_scores = hpsv2.score(chunk_images, chunk_prompts, hps_version="v2.0")
        scores.extend(chunk_scores)
    avg_score = sum(scores) / len(scores)
    # print(f"Average HPSv2 score: {avg_score}")
    return avg_score

def benchmark(
    models: List[str],
    dataset: Optional[str] = "ymhao/HPDv2",
    ds_split: Optional[str] = "test",
    ds_samples: Optional[int] = 100,
    batch_size: Optional[int] = 8,
):
    ds = load_dataset(dataset)[ds_split].shuffle().select(range(ds_samples))
    scores = {}
    for model in models:
        scores[model] = benchmark_one_model(model, ds, batch_size)

    print("=== Scores ===")
    for model in models:
        print(f"{model} avg score: {scores[model]}")

if __name__ == "__main__":
    typer.run(benchmark)
