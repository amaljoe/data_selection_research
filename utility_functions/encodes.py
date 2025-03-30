from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
from torch.nn import functional as F
import pickle
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

cache_dir = os.path.join(os.environ.get("CACHE_DIR", "./cache"), "utility")

def get_encodes_utility(prompts, references, dataset_name):
    utility_name = f'{dataset_name}_encodes'
    cache_file = os.path.join(cache_dir, f"{utility_name}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f'Utility: {utility_name} found in cache, loading from cache ✅')
        with open(cache_file, 'rb') as f:
            return pickle.load(f), utility_name
    print(f'Utility: {utility_name} not found in cache, computing now 🏃')
    utility = encode(prompts, references).to("cpu")
    utility = np.array(utility)
    with open(cache_file, 'wb') as f:
        pickle.dump(utility, f)
    # print(f'Utility: {utility_name} computed and saved to cache ✅')
    return utility, utility_name

def encode(prompts, references, embedding_model_name='BAAI/bge-large-en-v1.5'):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    tokenizer.max_subtokens_sequence_length = 512
    tokenizer.model_max_length = 512
    model = AutoModel.from_pretrained(embedding_model_name).to('cuda')
    model.eval()

    data_to_encode = [p + " " + r for p, r in zip(prompts, references)]

    encoded_input = tokenizer(data_to_encode, padding=True, truncation=True, return_tensors='pt').to(model.device)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    bs = 16
    output = []
    for i in tqdm(range(0, len(encoded_input['input_ids']), bs), desc='Encoding data to vectors'):
        output.extend(model(input_ids=input_ids[i:i+bs], attention_mask=attention_mask[i:i+bs]).pooler_output.detach())

    output = torch.stack(output)
    del tokenizer, model
    return output

if __name__=='__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from data_loader import get_mix_instruct

    prompts, references, ds_name = get_mix_instruct("train", 210)
    utility, utility_name = get_encodes_utility(prompts, references, ds_name)
    print(utility.shape)
