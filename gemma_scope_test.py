# from huggingface_hub import notebook_login
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import Counter
from sae_lens import SAE
from tqdm import tqdm
import argparse
import torch
import json
import csv
from transformer_lens import HookedTransformer
import transformer_lens.utils as utils

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

def sample_responses_and_evaluate(dataset, model, truth_judge, tokenizer_judge, target_layer):
    global i, result
    result = {}
    judge_predictions = []
    for i, example in tqdm(enumerate(dataset['train']), total=len(dataset['train'])):
        result[f"{i}"] = {
        "prompt": example['Question'], 
        "generation": "", 
        "step_by_step_non_zero_activations": [],
        "truth_label": None
        }
        with torch.no_grad():
            outputs = model.generate(example['Question'], max_new_tokens=50)
        result[f"{i}"]['generation'] = outputs
        
        prompt = f"Q: {example['Question']}?\nA: {outputs}\nTrue:"
        judge_inputs = tokenizer_judge.encode(prompt, return_tensors="pt").to(device_judge)
        outputs = truth_judge.generate(judge_inputs)
        pred_truth_label = tokenizer_judge.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        # print(pred_truth_label)

        result[f"{i}"]['truth_label'] = pred_truth_label
        with open(f"ongoing_{target_layer}.json", "w") as test:
            test.write(json.dumps(result))

    with open(f"data/activations/output_{target_layer}.json", "w") as output_json:
        output_json.write(json.dumps(result))
    
    predictions_out_path = f"data/predictions/gemma_predictions_truthfulQA_{target_layer}.csv"
    write_model_predictions(predictions_out_path, judge_predictions)
    output_json.close()
    
def SAE_on_layer_hook(value, hook):
    sae_acts = sae.encode(value.to(torch.float32))
    recon = sae.decode(sae_acts)

    if(len(sae_acts.shape) > 2):
        sae_acts = sae_acts.squeeze()
    if len(sae_acts.shape) == 1:
        sae_acts = sae_acts.unsqueeze(0) # uniforms to n_tokens, d_sae for all inputs (prompt included)
        
    for index, token in enumerate(sae_acts):
        indices = torch.nonzero(token, as_tuple=True)[0]
        values = token[indices]
        index_value_dict = dict(zip(indices.tolist(), values.tolist()))
        result[f"{i}"]['step_by_step_non_zero_activations'].append(index_value_dict)


def main():
    global sae, model, tokenizer_gemma
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--target_layer', type=int, required=True)
    args = parser.parse_args()

    model = HookedTransformer.from_pretrained_no_processing(
        "google/gemma-2-2b",
        device_map=device_gemma,
        torch_dtype=torch.float16
    ).to(device_gemma)
    print(model)
    
    tokenizer_gemma =  AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release = "gemma-scope-2b-pt-res-canonical",
        sae_id = f"layer_{args.target_layer}/width_16k/canonical",
    )
    sae.to(device_gemma)
    
    model.add_perma_hook(utils.get_act_name("resid_post", args.target_layer), SAE_on_layer_hook)
    
    truth_judge = AutoModelForCausalLM.from_pretrained(
        "allenai/truthfulqa-truth-judge-llama2-7B", 
        torch_dtype=torch.float16
        ).to(device_judge)
    tokenizer_judge = AutoTokenizer.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B")
    
    ds = load_dataset("domenicrosati/TruthfulQA")
    
    sample_responses_and_evaluate(ds, model, truth_judge, tokenizer_judge, args.target_layer)

if __name__ == "__main__":
    main()
