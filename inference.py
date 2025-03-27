from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pickle
import os
from dotenv import load_dotenv
load_dotenv()

cache_dir = os.path.join(os.environ.get("CACHE_DIR", "./cache"), "generated_texts")

def generate_responses(prompts, model_name, dataset_name, device='cuda:0', batch_size=8, max_length=512):
    model_name_short = model_name.split('/')[-1]
    generation_name = f'{dataset_name}_{model_name_short}'
    cache_file = os.path.join(cache_dir, f"{generation_name}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f'Generation: {generation_name} found in cache, loading from cache ✅')
        with open(cache_file, 'rb') as f:
            return pickle.load(f), generation_name
    print(f'Generation: {generation_name} not found in cache, generating responses 🏃')

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16).to(device)
    dataloader = DataLoader(prompts, batch_size=batch_size, shuffle=False)
    all_responses = []

    for batch in tqdm(dataloader, desc="Generating responses"):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_length, do_sample=False)
        responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        all_responses.extend(responses)

    with open(cache_file, 'wb') as f:
        pickle.dump(all_responses, f)
    print(f'Generation: {generation_name} generated and saved to cache ✅')
    return all_responses, generation_name