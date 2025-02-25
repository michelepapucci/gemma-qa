from scipy.spatial import distance
from tqdm import tqdm
import numpy as np
import json

def load_json(src_path):
    with open(src_path, 'r') as src_file:
        output = json.load(src_file)
    return output

def load_feature_activations(activations_dict, feature_idx):
    activations = {'yes':[], 'no':[]}
    for sentence_dict in activations_dict.values():
        sentence_activations_dict = sentence_dict['step_by_step_non_zero_activations']
        sentence_label = sentence_dict['truth_label']
        for token_activations in sentence_activations_dict:
            if feature_idx in token_activations:
                activations[sentence_label].append(token_activations[feature_idx])
    return activations        


def estimate_probability_distribution(data, bins='auto'):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    hist = hist / np.sum(hist)
    return hist, bin_edges

def main():
    activations_path = 'data/activations/output_20.json'
    num_components = 16384

    activations_dict = load_json(activations_path)
    activations_divergence = dict()
    for feature_idx in tqdm(range(1, num_components+1)):
        activations = load_feature_activations(activations_dict, str(feature_idx))
        no_distribution, bins = estimate_probability_distribution(np.array(activations['no']))
        yes_distribution, _ = estimate_probability_distribution(np.array(activations['yes']), bins)
        activations_divergence[feature_idx] = distance.jensenshannon(yes_distribution, no_distribution)

    with open('data/activations/divergence_20.json', 'w') as out_file:
        json.dump(activations_divergence, out_file)



if __name__ == '__main__':
    main()