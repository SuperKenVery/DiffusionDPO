import typer
from diffusers import AutoPipelineForText2Image
from datasets import load_dataset
import hpsv2
from typing import List, Optional, Union, Tuple, Callable

def extract_prompt(examples):
    return {"prompt": examples['prompt']}

def benchmark(
    model_path: str,
    dataset: Optional[str] = "ymhao/HPDv2",
    ds_split: Optional[str] = "test",
    ds_samples: Optional[int] = 100,
):
    model = AutoPipelineForText2Image.from_pretrained(model_path)
    ds = load_dataset(dataset)[ds_split].select(range(ds_samples))
    # .with_transform(extract_prompt)
    scores = []
    for sample in ds:
        prompt = sample['prompt']
        image = model(prompt, num_inference_steps=4).images[0]
        score = hpsv2.score(image, prompt, hps_version="v2.1")
        scores.append(score)
    avg_score = sum(scores) / len(scores)
    print(f"Average HPSv2 score: {avg_score}")
    return avg_score

if __name__ == "__main__":
    typer.run(benchmark)
