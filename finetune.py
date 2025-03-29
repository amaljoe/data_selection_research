import evaluate
from datasets import Dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, EvalPrediction, TrainingArguments, \
    TrainerCallback
import torch
from bert_score import score as bert_score
import numpy as np
from trl import SFTTrainer, SFTConfig
import os
from dotenv import load_dotenv
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
    fine_tune_model(base_model_id, s_prompts, s_references, prompts_val, references_val, ds_name, use_cache=False)

# {'eval_loss': 2.4013614654541016, 'eval_rouge1': 0.5915068179332093, 'eval_runtime': 17.7148, 'eval_samples_per_second': 2.822, 'eval_steps_per_second': 0.395, 'eval_mean_token_accuracy': 0.5173488073050976, 'epoch': 1.0}