from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle
import torch
from torch.nn import functional as F
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

cache_dir = os.path.join(os.environ.get("CACHE_DIR", "./cache"), "utility")

def get_delift_se_utility(prompts, references, dataset_name):
    utility_name = f'{dataset_name}_delift-se'
    cache_file = os.path.join(cache_dir, f"{utility_name}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f'Utility: {utility_name} found in cache, loading from cache ✅')
        with open(cache_file, 'rb') as f:
            return pickle.load(f), utility_name
    print(f'Utility: {utility_name} not found in cache, computing now 🏃')
    tensor = encode(prompts, references)
    utility =  compute_pairwise_similarities(tensor)
    utility = np.array(utility)
    with open(cache_file, 'wb') as f:
        pickle.dump(utility, f)
    print(f'Utility: {utility_name} computed and saved to cache ✅')
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

def compute_pairwise_similarities(tensor, metric='cosine', batch_size=10000, scaling=None, kw=0.1, device='cuda'):
    tensor = tensor.to(device)
    n_samples = tensor.size(0)

    # Initialize a results matrix in the CPU memory to save GPU memory
    results = torch.zeros(n_samples, n_samples, device='cpu')

    # Normalizing tensors if metric is cosine for cosine similarity computation
    if metric == 'cosine':
        tensor = F.normalize(tensor, p=2, dim=1)

    # Function to calculate the metric
    def calculate_metric(a, b, metric):
        if metric in ['cosine', 'dot']:
            return torch.mm(a, b.T)
        elif metric == 'euclidean':
            return torch.cdist(a, b, p=2)
        elif metric == 'rbf':
            distance = torch.cdist(a, b)
            squared_distance = distance ** 2
            avg_dist = torch.mean(squared_distance)
            torch.div(squared_distance, kw*avg_dist, out=squared_distance)
            torch.exp(-squared_distance, out=squared_distance)
            return squared_distance
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Process in batches to manage memory usage
    for i in tqdm(range(0, n_samples, batch_size), desc='Computing pairwise similarities'):
        end_i = min(i + batch_size, n_samples)
        rows = tensor[i:end_i]

        for j in range(0, n_samples, batch_size):
            end_j = min(j + batch_size, n_samples)
            cols = tensor[j:end_j]

            # Compute metric for the current batch and store results on CPU
            batch_results = calculate_metric(rows, cols, metric).to('cpu')
            results[i:end_i, j:end_j] = batch_results

    # Apply scaling if specified
    if scaling == 'min-max':
        min_val, max_val = results.min(), results.max()
        if max_val != min_val:
            results = (results - min_val) / (max_val - min_val)
    elif scaling == 'additive':
        results = (results + 1) / 2

    return results