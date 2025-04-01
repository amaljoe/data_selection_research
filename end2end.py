import random
import argparse
import pytz
import json
import os
from datetime import datetime
from dotenv import load_dotenv


from finetune_sft import fine_tune_model
from inference import generate_responses
from evaluation import compute_metrics
from data_loader import get_mix_instruct
from utility_functions.delift_se import get_delift_se_utility
from subset import create_subset, get_subset

load_dotenv()
cache_dir = os.path.join(os.environ.get("CACHE_DIR", "./cache"), "logs")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate an LLM model.")

    # Dataset parameters
    parser.add_argument("--train_seed", type=int, default=42, help="Random seed for training data selection")
    parser.add_argument("--train_length", type=int, default=21000, help="Number of training samples")
    parser.add_argument("--val_seed", type=int, default=42, help="Random seed for validation data selection")
    parser.add_argument("--val_length", type=int, default=1000, help="Number of validation samples")

    # Subset selection parameters
    parser.add_argument("--method", type=str, choices=['initial', 'random', 'delift-se', 'full'], default='initial',
                        help="Subset selection method")
    parser.add_argument("--subset_size", type=float, default=0.3, help="Proportion of data to keep in subset selection")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed for random subset selection")

    # Model parameters
    parser.add_argument("--model", type=str, default='meta-llama/Llama-3.2-3B', help="Base model identifier")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--generation_max_length", type=int, default=150, help="Maximum generation length during inference")
    parser.add_argument("--tag", type=str, default=None, help="Tag name to uniquely identify experiments")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    formatted_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d-%m-%Y-%H:%M:%S')
    experiment_name = f'{args.model}_{args.method}_{formatted_time}'
    if args.tag is not None:
        experiment_name += args.tag

    print(f'\nRunning experiment "{experiment_name}" with parameters:')
    max_key_length = max(len(str(k)) for k in vars(args).keys())
    for k, v in vars(args).items():
        print(f"{k:<{max_key_length}} : {v}")
    print("")

    prompts, references, ds_name = get_mix_instruct("train", args.train_length, seed=args.train_seed)
    if args.method == 'delift-se':
        utility, utility_name = get_delift_se_utility(prompts, references, ds_name)
        subset, subset_name = create_subset(utility, utility_name, k=args.subset_size)
        prompts, references = get_subset(subset, prompts, references)
    elif args.method == 'random':
        subset_indices = random.Random(args.random_seed).sample(range(len(prompts)), int(args.subset_size * len(prompts)))
        prompts = [prompts[i] for i in subset_indices]
        references = [references[i] for i in subset_indices]
    prompts_val, references_val, ds_name_val = get_mix_instruct("validation", args.val_length, seed=args.val_seed)
    model_dir = args.model
    if args.method != 'initial':
        model_dir = fine_tune_model(args.model, prompts, references, prompts_val, references_val, ds_name, epochs=args.epochs, tag=args.tag)
    responses, generation_name = generate_responses(prompts_val, model_dir, ds_name_val, max_length=args.generation_max_length)
    metrics = compute_metrics(responses, prompts_val, references_val, generation_name)

    results = {
        "experiment_name": experiment_name,
        "parameters": vars(args),
        "metrics": metrics
    }
    log_file = os.path.join(cache_dir, f'{experiment_name}.txt')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults for the experiment: {experiment_name}")
    max_key_length = max(len(str(k)) for k in metrics.keys())
    for k, v in metrics.items():
        print(f"{k:<{max_key_length}} : {v}")
