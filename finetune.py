import os
import random
from datetime import datetime

import pytz
import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, EvalPrediction, get_scheduler, \
    DataCollatorForLanguageModeling

from dynamic_weights import DynamicWeights

load_dotenv()

cache_dir = os.path.join(os.environ.get("CACHE_DIR", "./cache"), "models")

def formatting_prompts_func(prompts, references):
    return [
        f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

{prompt}

### Response:
{reference}
                """
        for prompt, reference in zip(prompts, references)
    ]

from torch.utils.data import DataLoader
from datasets import Dataset
import numpy as np

def get_data_loaders(prompts, references, subset, bs, formatting_fn, preprocess_fn, data_collator):
    prompts, references = np.array(prompts), np.array(references)
    domain_indices = []
    domains = []
    for s in subset:
        index, value = s[0], s[1]
        if value > 30:
            domain_indices.append([index])
            domains.append(index)
        else:
            d = np.argmax(utility[index, domains])
            domain_indices[d].append(index)

    data_loaders = []
    for idx in domain_indices:
        domain_prompts, domain_references = prompts[idx], references[idx]
        text = formatting_fn(domain_prompts, domain_references)
        ds = Dataset.from_dict({"text": text})
        ds = ds.map(preprocess_fn, batched=True)
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=data_collator)
        data_loaders.append(dl)

    # domain_prompts, domain_references = prompts, references
    # text = formatting_fn(domain_prompts, domain_references)
    # ds = Dataset.from_dict({"text": text})
    # ds = ds.map(preprocess_fn, batched=True)
    # ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    # dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=data_collator)
    return data_loaders

def fine_tune_odm(base_model_id, prompts, references, prompts_val, references_val, subset_name, use_cache=True):
    model_name = f"{base_model_id.split('/')[-1]}_{subset_name}"
    model_dir = os.path.join(cache_dir, model_name)

    os.makedirs(os.path.dirname(model_dir), exist_ok=True)

    if os.path.exists(model_dir) and use_cache:
        print(f"Finetune: Fine-tuned model found in cache. Skipping Training ✅")
        return model_dir
    elif os.path.exists(model_dir) and not use_cache:
        print(f"Finetune: Fine-tuned model found in cache. Invalidating cache and training now 🏃")
    else:
        print(f"Finetune: Fine-tuned model not found in cache. Training now 🏃")

    # Get current time in IST
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    formatted_time = current_time.strftime('%d-%m-%Y %H:%M:%S')
    writer = SummaryWriter(log_dir=os.path.join(model_dir, "runs", formatted_time))

    # Model setup
    quant_storage_dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=quant_storage_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=quant_storage_dtype,
        use_cache=False,
        device_map='auto'
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # PEFT Configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Prepare dataset
    valid_texts = formatting_prompts_func(prompts_val, references_val)
    valid_dataset = Dataset.from_dict({"text": valid_texts, "references": references_val})

    def preprocess_function(examples):
        inputs = tokenizer(examples["text"], truncation=True)
        return inputs

    def valid_preprocess_function(examples):
        tokenizer.padding_side = "left"
        inputs = tokenizer(examples["text"], truncation=True, padding=True, padding_side="left", max_length=200, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    valid_dataset = valid_dataset.map(preprocess_function, batched=True)
    valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])


    train_bs = 128
    mini_bs = 16
    valid_bs = 32
    eval_steps = 10
    num_epochs = 1



    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloaders = get_data_loaders(prompts, references, subset, mini_bs, formatting_prompts_func, preprocess_function, data_collator)

    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=valid_bs, collate_fn=data_collator)

    # Optimizer & Scheduler

    optimizer = AdamW(model.parameters(), lr=2.5e-5, weight_decay=0.01)
    num_batches = sum([len(d) for d in train_dataloaders])
    num_training_steps = num_batches * num_epochs
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Training Loop

    def train_step(batch):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        return loss.item()


    def validation():
        total_loss = 0.0
        num_tokens = 0

        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc='Validating...'):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item() * batch['input_ids'].numel()
                num_tokens += batch['input_ids'].numel()

        model.train()

        # Calculate average loss and perplexity
        avg_loss = total_loss / num_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))

        return perplexity



    dataloader_iters = [iter(dl) for dl in train_dataloaders]
    dynamic_weights = DynamicWeights([len(d) / num_batches for d in dataloader_iters])
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model.train()
        loop = tqdm(range(num_batches // (train_bs // mini_bs)), leave=True)
        for step in loop:
            if step % eval_steps == 0 or step == num_batches - 1:
                perplexity = validation()
                writer.add_scalar("eval/perplexity", perplexity, epoch * num_batches + step)
            mini_batch_loss = 0
            for mini_step in range(train_bs // mini_bs):
                try:
                    index = random.choices(np.arange(len(dataloader_iters)), weights=dynamic_weights.weights)[0]
                    batch = next(dataloader_iters[index])
                except StopIteration:
                    dataloader_iters[index] = iter(train_dataloaders[index])
                    batch = next(dataloader_iters[index])
                loss = train_step(batch)
                mini_batch_loss += loss
                dynamic_weights.update(index, loss, epoch * num_batches + step * (train_bs // mini_bs) + mini_step + 1)
            writer.add_scalar("train/loss", mini_batch_loss / (train_bs // mini_bs), epoch * num_batches + step)
            writer.add_scalar("train/epoch", epoch + step / num_batches, epoch * num_batches + step)
            for i, p in enumerate(dynamic_weights.weights):
                writer.add_scalar(f"weights/domain_{i}", p, epoch * num_batches + step)
            loop.set_description(f"Loss: {loss:.4f}")

    # Save model
    print("Training complete. Saving model.")
    writer.close()
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    del model
    torch.cuda.empty_cache()
    return model_dir


if __name__=='__main__':
    from data_loader import get_mix_instruct
    from utility_functions.delift_se import get_delift_se_utility
    from subset import create_subset, get_subset

    prompts, references, ds_name = get_mix_instruct("train", 210)
    utility, utility_name = get_delift_se_utility(prompts, references, ds_name)
    subset, subset_name = create_subset(utility, utility_name, k=1)
    s_prompts, s_references = get_subset(subset, prompts, references)

    prompts_val, references_val, ds_name_valid = get_mix_instruct("validation", 50)
    base_model_id = 'meta-llama/Llama-3.2-3B'
    # base_model_id = 'cache/models/Llama-3.2-3B_mix-instruct_train_21000_delift-se_0.3'
    fine_tune_odm(base_model_id, s_prompts, s_references, prompts_val, references_val, ds_name, use_cache=False)

# {'eval_loss': 2.4013614654541016, 'eval_rouge1': 0.5915068179332093, 'eval_runtime': 17.7148, 'eval_samples_per_second': 2.822, 'eval_steps_per_second': 0.395, 'eval_mean_token_accuracy': 0.5173488073050976, 'epoch': 1.0}