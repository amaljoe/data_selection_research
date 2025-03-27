from data_loader import get_mix_instruct
from inference import generate_responses
from evaluation import compute_metrics

prompts, references, ds_name = get_mix_instruct("train", 21000)
responses, generation_name = generate_responses(prompts, 'meta-llama/Llama-3.2-3B', ds_name, 'cuda:0', batch_size=64)
print(compute_metrics(responses, references, generation_name, device='cuda:0', metrics=['bge', 'rouge']))
