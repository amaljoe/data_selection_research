import os
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
import pickle
load_dotenv()

cache_dir = os.path.join(os.environ.get("CACHE_DIR", "./cache"), "datasets")

def get_mix_instruct(split, max_length, seed=42):
    ds_name = f"mix-instruct_{split}_{max_length}_{seed}"
    cache_file = os.path.join(cache_dir, f"{ds_name}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f'Dataset: {ds_name} found in cache, loading from cache ✅')
        ds = pd.read_pickle(cache_file)
    else:
        print(f'Dataset: {ds_name} not found in cache, downloading from huggingface 🏃')
        ds = load_dataset("llm-blender/mix-instruct")[split].shuffle(seed=seed).to_pandas()[:max_length]
        ds.to_pickle(cache_file)
        print(f'Dataset: {ds_name} loaded and saved to cache ✅')
    # instruction and input splitted by new line (space is used in paper)
    prompts = ds['instruction'] + "\n" + ds['input']
    references = ds['output']
    return list(prompts), list(references), ds_name




def get_mmlu(split, max_length=None, seed=42):
    def generate_prompt(question, choices):
        choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        return f"{question}\n{choices_str}\nAnswer:"

    ds_name = f"mmlu_{split}_{max_length}_{seed}"
    cache_file = os.path.join(cache_dir, f"{ds_name}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    prompts, references = [], []
    if os.path.exists(cache_file):
        print(f'Dataset: {ds_name} found in cache, loading from cache ✅')
        with open(cache_file, 'rb') as f:
            prompts, references = pickle.load(f)
    else:
        print(f'Dataset: {ds_name} not found in cache, downloading from huggingface 🏃')
        dataset = load_dataset("cais/mmlu", 'all', split=split)
        max_length = max_length if max_length is not None else len(dataset)
        dataset = dataset.shuffle(seed=42).select(range(max_length))
        prompts = [generate_prompt(example['question'], example['choices']) for example in dataset]
        references = [chr(65 + example['answer']) for example in dataset]
        with open(cache_file, 'wb') as f:
            pickle.dump((prompts, references), f)
        print(f'Dataset: {ds_name} loaded and saved to cache ✅')
    return prompts, references, ds_name

if __name__ == "__main__":
    prompts, references, ds_name = get_mix_instruct("train", 21000)
    print(f"Dataset: {ds_name} loaded with {len(prompts)} samples")