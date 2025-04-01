import torch.multiprocessing as mp
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
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

def evaluate_laj(predictions, prompts, references, return_individual=False, bs = 16):
    """
    Calculates the similarity between predictions and references using Prometheus (LLM-as-a-Judge).

    Args:
        predictions: list of strings for the hypothesis
        references: list of strings for the reference
        return_invidiual: if True, it will return the individual scores for corresponding prediction-reference pairs
    Returns:
        np array of metrics of size 1x1 if return_individual is True, else 1x|predictions|
    """
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    instructions = prompts
    responses = predictions

    rubric_data = {
        "criteria": """Evaluate the model's ability to follow instructions and deliver a high-quality response across the following dimensions:
1. **Instruction Following**: How accurately and fully does the model adhere to the given instruction?
2. **Accuracy**: Is the information correct, reliable, and factually sound?
3. **Relevance**: Does the response directly address the question or task without unnecessary information?
4. **Completeness**: Does the response cover all essential aspects of the instruction or question?
5. **Depth**: How thoroughly does the response explore the topic? Does it demonstrate insightful analysis where appropriate?
6. **Clarity**: Is the response well-organized, easy to follow, and free from ambiguity or confusion?
7. **Creativity**: Does the response offer original or innovative approaches where applicable?
8. **Helpfulness**: Does the response effectively meet the user's needs and provide value in solving the problem or addressing the query?""",

        "score1_description": "The response fails to meet expectations across most or all criteria. It does not follow the instruction, contains significant errors or misinformation, lacks relevance, is incomplete or shallow, unclear, unoriginal, and unhelpful.",

        "score2_description": "The response shows major deficiencies across several criteria. It partially follows the instruction but includes significant inaccuracies, is often irrelevant, incomplete, or lacks depth, clarity, creativity, and helpfulness.",

        "score3_description": "The response is average, meeting some but not all criteria. It follows the instruction but may fall short in terms of accuracy, depth, relevance, or helpfulness. Improvements in clarity and insightfulness may be needed.",

        "score4_description": "The response is strong, performing well across most criteria. It follows the instruction closely, is mostly accurate and relevant, provides good depth, and is well-structured. Minor improvements could enhance clarity, creativity, or helpfulness.",

        "score5_description": "The response excels in all or nearly all criteria. It fully follows the instruction, is highly accurate, directly relevant, complete, and demonstrates depth and insight. The response is well-organized, creative where appropriate, and very helpful in addressing the user's needs.",
    }

    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

    metrics = []
    # for i in tqdm(range(0, len(instructions), bs), desc='Evaluating LAJ'):
    #     batch_instructions = instructions[i: i + bs]
    #     batch_references = references[i: i + bs]
    #     batch_responses = responses[i: i + bs]
    #     feedback, score = judge.absolute_grade(
    #         instructions=batch_instructions,
    #         responses=batch_responses,
    #         rubric=score_rubric,
    #         reference_answers=batch_references
    #     )
    #
    #     metrics.append(score)
    feedback, score = judge.absolute_grade(
        instructions=instructions,
        responses=responses,
        rubric=score_rubric,
        reference_answers=references
    )
    metrics = score

    # clean up memory
    del model
    del judge
    torch.cuda.empty_cache()

    if return_individual:
        return np.array(metrics)
    else:
        return np.array(metrics).mean()

def compute_metrics(predictions, prompts, references, generation_name, device='cuda:0', bs_bge=512, bs_rouge=4096, metrics=['bge', 'rouge', 'laj'], use_cache=True):
    valid_metrics = set(['bge', 'rouge', 'laj'])
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
            elif metric == 'laj':
                metrics[metric] = evaluate_laj(predictions, prompts, references)
            with open(cache_file, 'wb') as f:
                pickle.dump(metrics[metric], f)
            print(f'Evaluate: {metric_name} computed and saved to cache ✅')
    return metrics

if __name__ == '__main__':
    from data_loader import get_mix_instruct
    from inference import generate_responses
    import random
    import argparse
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument('--tokenizer', type=str, default=None)
    args = parser.parse_args()
    model = args.model
    tokenizer = args.tokenizer
    print(model)
    prompts, references, ds_name = get_mix_instruct("validation", 5000)
    random.seed(42)  # Set seed for reproducibility
    # selected_indices = random.sample(range(len(prompts)), 5000)
    # prompts = [prompts[i] for i in selected_indices]
    # references = [references[i] for i in selected_indices]
    # 'meta-llama/Llama-3.2-3B' "cache/models/Llama-3.2-3B_mix-instruct_train_21000"
    responses, generation_name = generate_responses(prompts, model, ds_name, 'cuda:0', max_length=150, batch_size=64, tokenizer_name=tokenizer)
    print(compute_metrics(responses, prompts, references, generation_name, device='cuda:0', bs_bge=512, metrics=['rouge', 'bge', 'laj']))