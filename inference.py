from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pickle
import os
from dotenv import load_dotenv
load_dotenv()

cache_dir = os.path.join(os.environ.get("CACHE_DIR", "./cache"), "generated_texts")

def generate_responses(prompts, model_name, dataset_name, device='cuda:0', batch_size=8, max_length=100, use_cache=True):
    model_name_short = model_name.split('/')[-1]
    generation_name = f'{dataset_name}_{model_name_short}_{max_length}'
    cache_file = os.path.join(cache_dir, f"{generation_name}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file) and use_cache:
        print(f'Generation: {generation_name} found in cache, loading from cache ✅')
        with open(cache_file, 'rb') as f:
            return pickle.load(f), generation_name
    elif os.path.exists(cache_file) and not use_cache:
        print(f'Generation: {generation_name} found in cache. Invalidating and generating new responses 🏃')
    else:
        print(f'Generation: {generation_name} not found in cache, generating responses 🏃')

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).to(device)
    def formatting_prompts_func(prompts):
        return [
            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

{prompt}

### Response:
"""
            for prompt in prompts
        ]
    prompts = formatting_prompts_func(prompts)
    dataloader = DataLoader(prompts, batch_size=batch_size, shuffle=False)
    all_responses = []

    for batch in tqdm(dataloader, desc="Generating responses"):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_length, min_p=0.1, temperature=0.2)
        new_tokens = outputs[:, inputs.input_ids.shape[1]:]
        batch_gen_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        batch_gen_texts = [gen_text.replace("\n", " ") for gen_text in batch_gen_texts]
        all_responses.extend(batch_gen_texts)

    with open(cache_file, 'wb') as f:
        pickle.dump(all_responses, f)
    print(f'Generation: {generation_name} generated and saved to cache ✅')
    return all_responses, generation_name

if __name__ == '__main__':
    from data_loader import get_mix_instruct
    from inference import generate_responses
    from evaluation import compute_metrics

    device = 'cuda:0'

    prompts_val, references_val, ds_name_valid = get_mix_instruct("validation", 5000)
    # model_dir = 'cache/models/Llama-3.2-3B_mix-instruct_train_21000_delift-se_0.3'
    model_dir = 'meta-llama/Llama-3.2-3B'
    responses_val_llama_ft, generation_name_llama_ft = generate_responses(prompts_val, model_dir, ds_name_valid, device, batch_size=400, use_cache=False)
    compute_metrics(responses_val_llama_ft, references_val, generation_name_llama_ft, device=device)