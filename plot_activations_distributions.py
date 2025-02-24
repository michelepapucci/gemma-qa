import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import math
import os

def load_json(src_path):
    with open(src_path, 'r') as src_file:
        json_content = json.load(src_file)
    return json_content

def count_classes_predictions(predictions_path):
    predictions_df = pd.read_csv(predictions_path, index_col='index')
    predictions_count = predictions_df['prediction'].value_counts()
    return predictions_count.to_dict()

def vectorize_activations(activations, num_components):
    activations_vector = np.zeros(num_components)
    for idx, act in activations.items():
        activations_vector[int(idx)] = act
    return activations_vector

def vectorize_and_normalize_label_activations(label, activations, predictions_count, num_components):
    label_activations = activations[label]
    label_predictions_count = predictions_count[label]
    label_activations = vectorize_activations(label_activations, num_components)
    label_activations /= label_predictions_count
    return label_activations

def main():
    num_components = 16384
    num_layers = 26

    num_cols = 4
    num_rows = math.ceil(num_layers / num_cols)  # Calcola il numero di righe necessario
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4)) 
    axes = axes.flatten()

    for layer in range(num_components):
        activations_path = f'data/activations/output_{layer}.json'
        predictions_path = f'data/predictions/gemma_predictions_truthfulQA_{layer}.csv' 
        
        if not os.path.exists(activations_path):
            continue

        predictions_count = count_classes_predictions(predictions_path)
        activations = load_json(activations_path)

        yes_activations = vectorize_and_normalize_label_activations('yes', activations, predictions_count, num_components)
        no_activations = vectorize_and_normalize_label_activations('no', activations, predictions_count, num_components)

        ax = axes[layer]
        ax.scatter(x=yes_activations, y=no_activations)
        ax.set_title(f'Layer {layer}')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('data/results/activated_features.png', dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    main()