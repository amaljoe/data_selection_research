import os
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()

cache_dir = os.path.join(os.environ.get("CACHE_DIR", "./cache"), "datasets")

def get_mix_instruct(split, max_length):
    ds_name = f"mix-instruct_{split}_{max_length}"
    cache_file = os.path.join(cache_dir, f"{ds_name}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f'Dataset: {ds_name} found in cache, loading from cache ✅')
        ds = pd.read_pickle(cache_file)
    else:
        print(f'Dataset: {ds_name} not found in cache, downloading from huggingface 🏃')
        ds = load_dataset("llm-blender/mix-instruct")[split].to_pandas()[:max_length]
        ds.to_pickle(cache_file)
        print(f'Dataset: {ds_name} loaded and saved to cache ✅')
    # instruction and input splitted by new line (space is used in paper)
    prompts = ds['instruction'] + "\n" + ds['input']
    references = ds['output']
    return list(prompts), list(references), ds_name

if __name__ == "__main__":
    prompts, references, ds_name = get_mix_instruct("train", 21000)
    print(f"Dataset: {ds_name} loaded with {len(prompts)} samples")