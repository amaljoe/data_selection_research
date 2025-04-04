import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel
import torch
import json

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7

# Load models
def load_models():
    # Text generation model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config={"load_in_4bit": True} if DEVICE == "cuda" else None
    )

    # Evaluation models
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bge_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
    bge_model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')

    # LLM Judge
    judge_tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-7b-v2.0")
    judge_model = AutoModelForCausalLM.from_pretrained("prometheus-eval/prometheus-7b-v2.0").to(DEVICE)
    judge_pipe = pipeline("text-generation", model=judge_model, tokenizer=judge_tokenizer, device=DEVICE)

    return tokenizer, model, rouge, bge_tokenizer, bge_model, judge_pipe, judge_tokenizer

def format_prompt(instruction, input_text=None):
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{instruction}
{input_text if input_text else ''}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    return prompt

def generate_response(model, tokenizer, example):
    prompt = format_prompt(example['instruction'], example.get('input'))
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the original prompt from response
    return response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

def evaluate_model(dataset):
    # Load all models
    tokenizer, model, rouge, bge_tokenizer, bge_model, judge_pipe, judge_tokenizer = load_models()

    # Generate responses
    print("Generating responses...")
    for example in tqdm(dataset):
        example['generated'] = generate_response(model, tokenizer, example)
        example['reference'] = example['output']  # Alias for consistency

    # Calculate metrics
    print("\nCalculating ROUGE scores...")
    rouge_scores = calculate_rouge_scores(dataset, rouge)

    print("\nCalculating BGE scores...")
    bge_score = evaluate_bge(dataset, bge_tokenizer, bge_model)

    print("\nRunning LLM Judge evaluation...")
    llm_score = run_llm_judge(dataset, judge_pipe, judge_tokenizer)

    return {
        "rouge": rouge_scores,
        "bge": bge_score,
        "llm_judge": llm_score
    }

def calculate_rouge_scores(dataset, rouge):
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for example in dataset:
        score = rouge.score(example['reference'], example['generated'])
        for key in scores:
            scores[key].append(score[key].fmeasure)
    return {k: np.mean(v) for k, v in scores.items()}

def evaluate_bge(dataset, embedding_tokenizer, embedding_model, batch_size=16):
    embedding_model.to(DEVICE)
    embedding_model.eval()

    if embedding_tokenizer.pad_token is None:
        embedding_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        embedding_model.resize_token_embeddings(len(embedding_tokenizer))

    def find_embeddings(texts):
        encoded_input = embedding_tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model_output = embedding_model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        return torch.nn.functional.normalize(sentence_embeddings, dim=1)

    predictions = [ex['generated'] for ex in dataset]
    references = [ex['reference'] for ex in dataset]
    metrics = []
    for i in tqdm(range(0, len(predictions), batch_size), desc='Computing BGE'):
        batch_preds = predictions[i:i+batch_size]
        batch_refs = references[i:i+batch_size]

        pred_embs = find_embeddings(batch_preds)
        ref_embs = find_embeddings(batch_refs)

        scores = (ref_embs * pred_embs).sum(dim=1).cpu().numpy()
        metrics.extend(scores)

    return np.mean(metrics)

def create_judge_prompt(example):
    # Read JSON file and load it as a Python object
    with open('rubric_data.json', 'r') as file:
        rubric_data = json.load(file)

    return f"""###Task:
        Evaluate this response based on the criteria below:
        
        ###Instruction:
        {example['instruction']}
        
        ###Response:
        {example['generated']}
        
        ###Reference Answer:
        {example['reference']}
        
        ###Evaluation Criteria:
        {rubric_data['criteria']}
        
        ###Scoring Guide:
        1: {rubric_data['score1_description']}
        2: {rubric_data['score2_description']}
        3: {rubric_data['score3_description']}
        4: {rubric_data['score4_description']}
        5: {rubric_data['score5_description']}
        
        First, provide a detailed analysis following the criteria. Conclude with your final score using the format: [RESULT] [X] where X is the score."""

def run_llm_judge(dataset, judge_pipe, judge_tokenizer):
    scores = []
    for example in dataset:
        prompt = create_judge_prompt(example)
        response = judge_pipe(
            prompt,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.1,
            pad_token_id=judge_tokenizer.eos_token_id
        )[0]['generated_text']

        # Extract score from response
        score_str = response.split('[RESULT]')[-1].strip()
        try:
            score = int(score_str[1:-1])
            scores.append(score)
        except:
            print(f"Error parsing score from: {score_str}")
            scores.append(0)
    return np.mean(scores)

def main():
    dataset = load_dataset("llm-blender/mix-instruct")["validation"]
    dataset = [
        {
            'instruction': ex['instruction'],
            'input': ex['input'],
            'output': ex['output']  # This will be our reference
        }
        for ex in dataset
    ]
    results = evaluate_model(dataset[:5])

    print("\nFinal Evaluation Results:")
    print(f"ROUGE Scores: {results['rouge']}")
    print(f"BGE Cosine Similarity: {results['bge']:.4f}")
    print(f"LLM Judge Average Score: {results['llm_judge']:.2f}/5")

if __name__ == "__main__":
    main()