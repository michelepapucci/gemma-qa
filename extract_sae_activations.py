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

device_gemma = "cuda:0"
torch.set_float32_matmul_precision('high')
torch.set_grad_enabled(False) # avoid blowing up mem
model_name = "google/gemma-2-2b"

def gather_residual_activations(model, target_layer, inputs):
  target_act = None
  def gather_target_act_hook(mod, inputs, outputs):
    nonlocal target_act # make sure we can modify the target_act from the outer scope
    target_act = outputs[0]
    return outputs
  handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
  _ = model.forward(inputs)
  handle.remove()
  return target_act

def load_model_predictions(src_path):
    predictions_dict = {}
    with open(src_path, 'r') as src_file:
        csv_reader = csv.reader(src_file)
        next(csv_reader) # skip the header
        for row in csv_reader:
            predictions_dict[row[0]] = row[1]
    return predictions_dict

def sample_responses_and_evaluate(dataset, predictions_dict, model, sae, tokenizer_gemma, target_layer):
    judge_predictions = []
    output_dict = {"yes":{}, "no":{}}

    for i, example in tqdm(enumerate(dataset['train']), total=len(dataset['train'])):
        inputs = tokenizer_gemma.encode(example['Question'], return_tensors="pt", add_special_tokens=True).to(device_gemma)

        target_act = gather_residual_activations(model, target_layer, inputs)
        sae_acts = sae.encode(target_act.to(torch.float32))
        # recon = sae.decode(sae_acts)

        squeezed_sae_acts = sae_acts.squeeze()

        significant_activations = []
        for index, token in enumerate(squeezed_sae_acts):
            significant_activations += [el.item() for el in token.nonzero()]
        frequency_sorted_activations = dict(Counter(significant_activations))
        frequency_sorted_activations = dict(sorted(frequency_sorted_activations.items(), key=lambda item: item[1], reverse=True))

        pred_truth_label = predictions_dict[i]

        if pred_truth_label in output_dict:
            for feature in frequency_sorted_activations:
                if feature not in output_dict[pred_truth_label]:
                    output_dict[pred_truth_label][feature] = 0
                output_dict[pred_truth_label][feature] += 1
        judge_predictions.append((i, pred_truth_label))

    with open(f"data/activations/output_{target_layer}.json", "w") as output_json:
        output_json.write(json.dumps(output_dict))
    
    output_json.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--target_layer', type=int, required=True)
    args = parser.parse_args()

    predictions_path = 'data/predictions/gemma_predictions_truthfulQA.csv'
    predictions_dict = load_model_predictions(predictions_path)

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        device_map=device_gemma,
        torch_dtype=torch.float16
    )
    model.to(device_gemma)
    tokenizer_gemma =  AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release = "gemma-scope-2b-pt-res-canonical",
        sae_id = f"layer_{args.target_layer}/width_16k/canonical",
    )
    sae.to(device_gemma)

    
    dataset = load_dataset("domenicrosati/TruthfulQA")
    
    sample_responses_and_evaluate(dataset, predictions_dict, model, sae, tokenizer_gemma, args.target_layer)

if __name__ == "__main__":
    main()
