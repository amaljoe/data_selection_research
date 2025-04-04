from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import random
from datasets import load_dataset

# Load the MMLU dataset
dataset = load_dataset("cais/mmlu", 'all', split='test')


def generate_prompt(question, choices):
    choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    return f"{question}\n{choices_str}\nAnswer:"

def generate_prompt_shot(question, choices, examples):
    # Create 5-shot prompt
    shot_text = ""
    for ex in examples:
        example_text = f"Q: {ex['question']}\n"
        example_text += "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(ex['choices'])])
        example_text += f"\nAnswer: {chr(65 + ex['answer'])}\n\n"
        shot_text += example_text

    # Create the actual prompt
    question_text = f"Q: {question}\n"
    question_text += "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    question_text += "\nAnswer:"

    return shot_text + question_text

def evaluate_mmlu(model, tokenizer, dataset):
    correct = 0
    total = 0

    dataset = dataset.shuffle().select(range(len(dataset)))

    loop = tqdm(enumerate(dataset), total=len(dataset))
    for i, example in loop:
        if i < 5:
            continue
        indices = range(i - 5, i)
        examples = dataset.select(indices)
        answer = chr(65 + example['answer'])  # Convert index to A, B, C, or D

        prompt = generate_prompt_shot(example['question'], example['choices'], examples)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
            pred = tokenizer.decode(outputs[0][-1], skip_special_tokens=True).strip().upper()

        is_correct = pred == answer
        correct += is_correct
        total += 1
        loop.set_description(f'{correct / total:.4f}')

    print(f"Final Accuracy: {correct / total:.4f}")

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B")
    model_name = parser.parse_args().model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    evaluate_mmlu(model, tokenizer, dataset)
