import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import argparse

def plot_layer_contribution_heatmap(checkpoint_path, output_path=None):
    # Load the model state dictionary
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    weights = []
    
    # extract and process weights for each layer
    for i in range(12): 
        alpha_weights_key = f'bert.transformer.layers.{i}.prev_layer_weights'
        
        if alpha_weights_key in state_dict:
            alpha_weights = state_dict[alpha_weights_key]
            softmax_alpha_weights = F.softmax(alpha_weights, dim=-1)
            weights.append(softmax_alpha_weights.tolist())
        else:
            print(f"Key {alpha_weights_key} not found in state_dict")

    normalised_weight = []
    
    # normalize each layer's weights so they sum to 1 (proportions)
    for layer_weights in weights:
        total = sum(layer_weights)
        if total > 0:
            proportions = [w / total for w in layer_weights]
        else:
            proportions = layer_weights
        normalised_weight.append(proportions)

    max_length = 12

    # padding to the weights
    padded_weights = []
    for weight_list in normalised_weight:
        padding = [0] * (max_length - len(weight_list))
        padded_sublist = weight_list + padding
        padded_weights.append(padded_sublist)

    weights_array = np.array(padded_weights) * 100  # convert to percentages

    # mask upper triangle
    mask = np.triu(np.ones_like(weights_array, dtype=bool), k=1)

    # create the heatmap plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights_array, mask=mask, cmap='Blues', square=True, annot=True, fmt=".1f",
                cbar_kws={"shrink": .8, "label": "Contribution (%)"}, linewidths=0,
                xticklabels=np.arange(0, 12), yticklabels=np.arange(1, 13))

    plt.xlabel('Layer Outputs', fontsize=14, color='black')
    plt.ylabel('Layer Inputs', fontsize=14, color='black')
    plt.xticks(color='black')
    plt.yticks(color='black')
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')

    # save img as a png file
    if output_path is None:
        output_path = checkpoint_path.split('/')[-1].replace('.bin', '_proportions.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight', transparent=False)

    plt.show()
    print(f"Heatmap saved as {output_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate layer contribution heatmap from a model checkpoint.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the model checkpoint (e.g., 'eim/200.bin')")
    parser.add_argument('--output_path', type=str)
    
    args = parser.parse_args()
    
    plot_layer_contribution_heatmap(args.input_path, args.output_path)