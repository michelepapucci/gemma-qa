# from huggingface_hub import notebook_login
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch
import csv

device_gemma = "cuda:0"
device_judge = "cuda:1"
torch.set_float32_matmul_precision('high')
torch.set_grad_enabled(False) # avoid blowing up mem
model_name = "google/gemma-2-2b"


def write_model_predictions(output_path, judge_predictions):
    with open(output_path, "w") as output_csv: 
        writer = csv.writer(output_csv)
        writer.writerow(["index", "prediction"])
        for prediction in judge_predictions:
            writer.writerow(prediction)

def sample_responses_and_evaluate(dataset, model, tokenizer_gemma, truth_judge, tokenizer_judge, predictions_out_path):
    judge_predictions = []

    for i, example in tqdm(enumerate(dataset['train']), total=len(dataset['train'])):
        inputs = tokenizer_gemma.encode(example['Question'], return_tensors="pt", add_special_tokens=True).to(device_gemma)

        with torch.no_grad():
            outputs = model.generate(input_ids=inputs, max_new_tokens=50)
        gemma_output = tokenizer_gemma.decode(outputs[0])
        
        prompt = f"Q: {example['Question']}?\nA: {gemma_output}\nTrue:"
        judge_inputs = tokenizer_judge.encode(prompt, return_tensors="pt").to(device_judge)
        outputs = truth_judge.generate(judge_inputs)
        pred_truth_label = tokenizer_judge.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

        judge_predictions.append((i, pred_truth_label))
        
    write_model_predictions(predictions_out_path, judge_predictions)

def main():
    # judge_model = "allenai/truthfulqa-truth-judge-llama2-7B"
    judge_model = "allenai/truthfulqa-info-judge-llama2-7B"
    output_path = "data/predictions/gemma_predictions_truthfulQA_info_judge.csv"

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        device_map=device_gemma,
        torch_dtype=torch.float16
    )
    model.to(device_gemma)
    tokenizer_gemma =  AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    truth_judge = AutoModelForCausalLM.from_pretrained(
        judge_model, 
        torch_dtype=torch.float16
        ).to(device_judge)
    
    tokenizer_judge = AutoTokenizer.from_pretrained(judge_model)
    
    dataset = load_dataset("domenicrosati/TruthfulQA")
    
    sample_responses_and_evaluate(dataset, model, tokenizer_gemma, truth_judge, tokenizer_judge, output_path)

if __name__ == "__main__":
    main()
