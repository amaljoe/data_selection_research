from transformers import AutoTokenizer, AutoModel
import torch
import evaluate
from tqdm import tqdm
import pickle
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

cache_dir = os.path.join(os.environ.get("CACHE_DIR", "./cache"), "metrics")

def evaluate_rouge(predictions, references, batch_size):
    rouge_metric = evaluate.load('rouge')
    scores = []

    for i in tqdm(range(0, len(predictions), batch_size), desc="Evaluating ROUGE"):
        batch_predictions = predictions[i:i+batch_size]
        batch_references = references[i:i+batch_size]
        batch_scores = rouge_metric.compute(predictions=batch_predictions, references=batch_references)['rouge1']
        scores.append(batch_scores)
    return np.mean(scores)

def evaluate_bge(predictions, references, device, batch_size):
    embedding_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
    embedding_model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
    embedding_model.to(device)
    embedding_model.eval()

    if embedding_tokenizer.pad_token is None:
        embedding_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        embedding_model.resize_token_embeddings(len(embedding_tokenizer))

    def find_embeddings(texts):
        encoded_input = embedding_tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = embedding_model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        return torch.nn.functional.normalize(sentence_embeddings, dim=1)

    metrics = []
    for i in tqdm(range(0, len(predictions), batch_size), desc='Computing BGE'):
        batch_preds = predictions[i:i+batch_size]
        batch_refs = references[i:i+batch_size]

        pred_embs = find_embeddings(batch_preds)
        ref_embs = find_embeddings(batch_refs)

        scores = (ref_embs * pred_embs).sum(dim=1).cpu().numpy()
        metrics.extend(scores)

    return np.mean(metrics)


def compute_metrics(predictions, references, generation_name, device='cuda:0', bs_bge=512, bs_rouge=4096, metrics=['bge', 'rouge'], use_cache=True):
    valid_metrics = set(['bge', 'rouge'])
    metrics_dict = {}
    for metric in metrics:
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}")
        else:
            metrics_dict[metric] = None
    metrics = metrics_dict
    for metric in metrics.keys():
        metric_name = f'{generation_name}_{metric}'
        cache_file = os.path.join(cache_dir, f"{metric_name}.pkl")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if os.path.exists(cache_file) and use_cache:
            print(f'Evaluate: {metric_name} found in cache, loading from cache ✅')
            with open(cache_file, 'rb') as f:
                metrics[metric] = pickle.load(f)
        else:
            if os.path.exists(cache_file):
                print(f'Evaluate: {metric_name} found in cache. Invalidating cache and recomputing now 🏃')
            else:
                print(f'Evaluate: {metric_name} not found in cache, computing now 🏃')
            if metric == 'rouge':
                metrics[metric] = evaluate_rouge(predictions, references, bs_rouge)
            elif metric == 'bge':
                metrics[metric] = evaluate_bge(predictions, references, device, bs_bge)
            with open(cache_file, 'wb') as f:
                pickle.dump(metrics[metric], f)
            print(f'Evaluate: {metric_name} computed and saved to cache ✅')
    return metrics

if __name__ == '__main__':
    from data_loader import get_mix_instruct
    from inference import generate_responses
    prompts, references, ds_name = get_mix_instruct("validation", 50)
    # 'meta-llama/Llama-3.2-3B' "cache/models/Llama-3.2-3B_mix-instruct_train_21000"
    responses, generation_name = generate_responses(prompts, 'meta-llama/Llama-3.2-3B', ds_name, 'cuda:0', batch_size=64)
    print(compute_metrics(responses, references, generation_name, device='cuda:0', bs_bge=512))