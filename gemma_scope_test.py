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
device_judge = "cuda:1"
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


def write_model_predictions(output_path, judge_predictions):
    with open(output_path, "w") as output_csv: 
        writer = csv.writer(output_csv)
        writer.writerow("index, prediction")
        for prediction in judge_predictions:
            writer.writerow(prediction)

def sample_responses_and_evaluate(dataset, model, sae, tokenizer_gemma, truth_judge, tokenizer_judge, target_layer):
    judge_predictions = []
    output_dict = {"yes":{}, "no":{}}
    for i, example in tqdm(enumerate(dataset['train']), total=len(dataset['train'])):
        inputs = tokenizer_gemma.encode(example['Question'], return_tensors="pt", add_special_tokens=True).to(device_gemma)
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs, max_new_tokens=50)
        gemma_output = tokenizer_gemma.decode(outputs[0])
        
        target_act = gather_residual_activations(model, target_layer, inputs)
        sae_acts = sae.encode(target_act.to(torch.float32))
        recon = sae.decode(sae_acts)

        # print('Explained Var:', 1 - torch.mean((recon[:, 1:] - target_act[:, 1:].to(torch.float32)) **2) / (target_act[:, 1:].to(torch.float32).var()))
        squeezed_sae_acts = sae_acts.squeeze()
        # mean = (squeezed_sae_acts[squeezed_sae_acts > 0]).float().mean()  # Use .float() to avoid integer overflow
        # std = (squeezed_sae_acts[squeezed_sae_acts > 0]).float().std()
        # print(mean.item(), std.item())
        significant_activations = []
        for index, token in enumerate(squeezed_sae_acts):
            # if(index == 0):
            #     continue
            significant_activations += [el.item() for el in token.nonzero()]
        frequency_sorted_activations = dict(Counter(significant_activations))
        frequency_sorted_activations = dict(sorted(frequency_sorted_activations.items(), key=lambda item: item[1], reverse=True))

        prompt = f"Q: {example['Question']}?\nA: {gemma_output}\nTrue:"
        judge_inputs = tokenizer_judge.encode(prompt, return_tensors="pt").to(device_judge)
        outputs = truth_judge.generate(judge_inputs)
        pred_truth_label = tokenizer_judge.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        # print(pred_truth_label)

        if pred_truth_label in output_dict:
            for feature in frequency_sorted_activations:
                if feature not in output_dict[pred_truth_label]:
                    output_dict[pred_truth_label][feature] = 0
                output_dict[pred_truth_label][feature] += 1
        judge_predictions.append((i, pred_truth_label))

    with open(f"output_{target_layer}.json", "w") as output_json:
        output_json.write(json.dumps(output_dict))
    
    predictions_out_path = f"gemma_predictions_truthfulQA_{target_layer}.csv"
    write_model_predictions(predictions_out_path, judge_predictions)
    output_json.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--target_layer', type=int, required=True)
    args = parser.parse_args()

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
    
    truth_judge = AutoModelForCausalLM.from_pretrained(
        "allenai/truthfulqa-truth-judge-llama2-7B", 
        torch_dtype=torch.float16
        ).to(device_judge)
    tokenizer_judge = AutoTokenizer.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B")
    
    ds = load_dataset("domenicrosati/TruthfulQA")
    
    sample_responses_and_evaluate(ds, model, sae, tokenizer_gemma, truth_judge, tokenizer_judge, args.target_layer)

if __name__ == "__main__":
    main()
