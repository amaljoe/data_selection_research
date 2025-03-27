from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel
import pickle
import os
from dotenv import load_dotenv
load_dotenv()

cache_dir = os.path.join(os.environ.get("CACHE_DIR", "./cache"), "generated_texts")

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    print(f"Setting up distributed GPU {torch.cuda.get_device_name(local_rank)}({device}) as rank {dist.get_rank()}")
    return device

def cleanup_distributed():
    dist.destroy_process_group()


def generate_responses(prompts, model_name, dataset_name, batch_size=8, max_length=512):
    device = setup_distributed()
    print()
    model_name_short = model_name.split('/')[-1]
    generation_name = f'{dataset_name}_{model_name_short}'
    cache_file = os.path.join(cache_dir, f"{generation_name}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if dist.get_rank() == 0 and os.path.exists(cache_file):
        print(f'Generation: {generation_name} found in cache, loading from cache ✅ {dist.get_rank()}')
        cleanup_distributed()
        with open(cache_file, 'rb') as f:
            return pickle.load(f), generation_name
    if dist.get_rank() == 0:
        print(f'Generation: {generation_name} not found in cache, generating responses 🏃')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16).to(device)
    model = DataParallel(model)
    sampler = DistributedSampler(prompts, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    dataloader = DataLoader(prompts, batch_size=batch_size, sampler=sampler)
    all_responses = []

    for batch in tqdm(dataloader, desc=f"Generating responses on GPU {dist.get_rank()}"):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.module.generate(**inputs, max_new_tokens=max_length, do_sample=False)
        responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        all_responses.extend(responses)

    all_gathered_responses = [None] * dist.get_world_size()
    dist.all_gather_object(all_gathered_responses, all_responses)
    if dist.get_rank() == 0:
        final_responses = sum(all_gathered_responses, [])
        with open(cache_file, 'wb') as f:
            pickle.dump(final_responses, f)
        print(f'Generation: {generation_name} generated and saved to cache ✅')
        cleanup_distributed()
        return final_responses, generation_name

if __name__=='__main__':
    print("Running parallel inference")
    from data_loader import get_mix_instruct
    prompts, references, ds_name = get_mix_instruct("train", 1000)
    model_name = "microsoft/Phi-3-mini-128k-instruct"
    responses, generation_name = generate_responses(prompts[:32], model_name, ds_name + "_32")
