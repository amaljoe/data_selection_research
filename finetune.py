import os
from datetime import datetime
import evaluate
import numpy as np
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, EvalPrediction, get_scheduler, \
    DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig

load_dotenv()

cache_dir = os.path.join(os.environ.get("CACHE_DIR", "./cache"), "models")

# class ProfCallback(TrainerCallback):
#     def __init__(self, prof):
#         self.prof = prof
#
#     def on_step_end(self, args, state, control, **kwargs):
#         self.prof.step()


def fine_tune_model(base_model_id, prompts, references, prompts_val, references_val, subset_name, use_cache=True):
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

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"----------using {num_gpus}*GPUs----------")
        model = torch.nn.DataParallel(model).module

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Assuming `tokenizer` is already defined and available
    def compute_metrics(p: EvalPrediction):
        logits, labels = p
        logits = np.argmax(logits, axis=-1)
        predictions = tokenizer.batch_decode(logits, skip_special_tokens=True)
        import pickle
        with open("cache/misc/pred_val.pkl", "wb") as f:
            pickle.dump(predictions, f)
        labels[labels < 0] = tokenizer.eos_token_id
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute ROUGE scores
        rouge = evaluate.load('rouge')
        rouge_scores = rouge.compute(predictions=predictions, references=references)

        # Return combined metrics
        return {
            'rouge1': rouge_scores['rouge1']
        }

    def formatting_prompts_func(prompts, references):
        return [
            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

{prompt}

### Response:
{reference}
                """
            for prompt, reference in zip(prompts, references)
        ]

    train_dataset = Dataset.from_dict({
        "text": formatting_prompts_func(prompts, references)
    })

    valid_dataset = Dataset.from_dict({
        "text": formatting_prompts_func(prompts_val, references_val),
    })

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )

    max_seq_length = 1024

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        with_stack=True)

    sft_config = SFTConfig(
        max_seq_length=max_seq_length,
        packing=True,
        eval_packing=False,
        dataset_text_field="text",
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
        output_dir=model_dir,
        num_train_epochs=3,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        learning_rate=2.5e-5,
        bf16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",
        lr_scheduler_type="constant",
        weight_decay=0.01,
        report_to="tensorboard",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':True},
        load_best_model_at_end=True,  # Load the best model at the end for inference
        metric_for_best_model="rouge1",  # Choose based on the evaluation metric
        greater_is_better=True,
    )


    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        args=sft_config,
    )

    # trainer.add_callback(ProfCallback(prof))

    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################
    # prof.start()
    trainer.train()
    # prof.stop()

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    if hasattr(trainer, 'is_fsdp_enabled') and trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    print(f"Finetune: Model fine-tuning completed and saved to cache ✅")

    del model
    torch.cuda.empty_cache()
    return model_dir

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
    return data_loaders



def fine_tune_loop(base_model_id, prompts, references, prompts_val, references_val, subset_name, use_cache=True):
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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=os.path.join(model_dir, "runs", timestamp))

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
    train_texts = formatting_prompts_func(prompts, references)
    valid_texts = formatting_prompts_func(prompts_val, references_val)

    train_dataset = Dataset.from_dict({"text": train_texts})
    valid_dataset = Dataset.from_dict({"text": valid_texts, "references": references_val})

    def preprocess_function(examples):
        inputs = tokenizer(examples["text"], truncation=True)
        return inputs

    def valid_preprocess_function(examples):
        inputs = tokenizer(examples["text"], truncation=True, padding=True, padding_side="left", max_length=200)
        return inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    valid_dataset = valid_dataset.map(valid_preprocess_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    train_bs = 128
    valid_bs = 32
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(train_dataset, batch_size=train_bs, shuffle=False, collate_fn=data_collator)
    tokenizer.padding_side = "left"
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_bs)


    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=2.5e-5, weight_decay=0.01)
    num_training_steps = len(train_dataloader) * 3
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rouge = evaluate.load('rouge')


    # Training Loop
    num_epochs = 1
    eval_steps = 50

    def train_step(batch, step):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        writer.add_scalar("train/loss", loss.item(), epoch * len(train_dataloader) + step)
        writer.add_scalar("train/epoch", epoch + step / len(train_dataloader), epoch * len(train_dataloader) + step)
        return loss.item()

    def validation(step):
        model.eval()
        predictions, references = [], []
        with torch.no_grad():
            for batch_valid in tqdm(valid_dataloader, desc='Validating...'):
                batch_valid = {k: v.to(device) if k in ['input_ids', 'attention_mask', 'labels'] else v for k, v in batch_valid.items()}
                outputs_valid = model.generate(input_ids=batch_valid['input_ids'], attention_mask=batch_valid['attention_mask'], max_new_tokens=128)
                decoded_preds = tokenizer.batch_decode(outputs_valid, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(batch_valid['references'])
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        writer.add_scalar("eval/rouge1", rouge_scores['rouge1'], epoch * len(train_dataloader) + step)
        print(f"Validation ROUGE-1: {rouge_scores['rouge1']:.4f}")
        model.train()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model.train()
        total_loss = 0
        loop = tqdm(train_dataloader, leave=True)
        for step, batch in enumerate(loop):
            if step % eval_steps == 0:
                validation(step)
            loss = train_step(batch, step)
            total_loss += loss
            loop.set_description(f"Loss: {loss:.4f}")
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

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

    prompts, references, ds_name = get_mix_instruct("train", 21000)
    utility, utility_name = get_delift_se_utility(prompts, references, ds_name)
    subset, subset_name = create_subset(utility, utility_name, k=1)
    s_prompts, s_references = get_subset(subset, prompts, references)

    prompts_val, references_val, ds_name_valid = get_mix_instruct("validation", 50)
    base_model_id = 'meta-llama/Llama-3.2-3B'
    # base_model_id = 'cache/models/Llama-3.2-3B_mix-instruct_train_21000_delift-se_0.3'
    fine_tune_loop(base_model_id, s_prompts, s_references, prompts_val, references_val, ds_name, use_cache=False)

# {'eval_loss': 2.4013614654541016, 'eval_rouge1': 0.5915068179332093, 'eval_runtime': 17.7148, 'eval_samples_per_second': 2.822, 'eval_steps_per_second': 0.395, 'eval_mean_token_accuracy': 0.5173488073050976, 'epoch': 1.0}