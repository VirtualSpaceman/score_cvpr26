import numpy as np
import torch
from pprint import pprint
import json 
import os
import copy 
from src.merging.task_vectors import TaskVector
from src.args import parse_arguments
from src.constants import get_dict_dataset_paths, get_dict_epochs
from tqdm import tqdm 
import math

import matplotlib.pyplot as plt


# Config
args = parse_arguments()
pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'



def compute_pairwise_avg_matrix(allignment_ratios_pairs, datasets):
    """
    Build an (N x N) matrix where N = len(datasets).
    Rows = source datasets (current_ds), Columns = target datasets (next_ds).
    Each cell = mean of the list in allignment_ratios_pairs[source][target], or np.nan for empty.
    """
    n = len(datasets)
    mat = np.full((n, n), np.nan, dtype=float)

    for i, src in enumerate(datasets):
        row = allignment_ratios_pairs.get(src, {})
        for j, tgt in enumerate(datasets):
            vals = row.get(tgt, [])
            if len(vals) > 0:
                mat[i, j] = round(float(np.mean(vals)), 3)

    return mat


def plot_pairwise_alignment_matrix(mat, datasets, title=None, save_path=None, vmax=1.0, vmin=0.0):
    """
    Plot the pairwise matrix 'mat' as a heatmap.
    - mat: 2D numpy array (shape NxN), with np.nan for missing pairs.
    - datasets: list of labels (length N) for ticks.
    Returns the matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 0.5), max(6, len(datasets) * 0.5)))
    # Show the matrix; use default colormap; constrain to [vmin, vmax]
    # NaNs will appear as a different color (matplotlib handles this)
    FONT_SIZE = 17
    cmap = None  # use default colormap
    cmap = 'magma'
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)

    # colorbar
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label('Average alignment ratio')

    # ticks
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_yticklabels(datasets, fontsize=FONT_SIZE)

    ax.set_title(title or 'Pairwise average alignment ratios (source → target)', fontsize=FONT_SIZE)

    # annotate cells with numeric values where available
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                txt = ''
            else:
                txt = f"{val:.2f}"
            # choose label color for contrast
            if np.isnan(val):
                text_color = 'black'
            else:
                text_color = 'black' if val >= 0.85 else 'white'
            ax.text(j, i, txt, ha='center', va='center', color=text_color, fontsize=13)

    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(rotation=45, ha='right', fontsize=FONT_SIZE)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        print(f"Saved heatmap to: {save_path}")

    return fig


def calc_rank(S, norm_thresh=0.95):
    # Rank based on approximation error (Eq. 6) in the paper
    rank = np.argmax(np.sqrt(np.cumsum(S.pow(2) / S.pow(2).sum())) > norm_thresh)
    return rank


def alignment_ratio(S, S_proj):
    # Subspace alignment ratio based on norms of projected task matrix vs norm of the original one (Eq. 5) in the paper
    return np.linalg.norm(S_proj, ord=2) / np.linalg.norm(S, ord=2)


if __name__ == '__main__':    
    PATHS = get_dict_dataset_paths()
    EPOCHS = get_dict_epochs()
    datasets = ['Cars', 'MNIST', 'EuroSAT', 'SVHN', 'RESISC45', 'SUN397', 'DTD', 'GTSRB']
    epochs = {
        'Cars': 35,
        'DTD': 75,
        'EuroSAT': 12,
        'GTSRB': 11,
        'MNIST': 5,
        'RESISC45': 15,
        'SUN397': 14,
        'SVHN': 4,
        'ImageNet': 4
    }
    args.eval_datasets = datasets
    args.lr = 1e-5
    args.batch_size = 128
    method = 'seq-ft' if args.sequential_finetuning else 'ind-ft'

    base_path = f'checkpoints/{args.model}/8datasets/ind'

    task_vectors_dict = dict()
    for ds in datasets:
        ft_path = os.path.join(base_path, f'{ds}/checkpoint-epochs:{epochs[ds]}-seed:{args.seed}.pt')
        tv = TaskVector(pretrained_checkpoint, ft_path)
        task_vectors_dict[ds] = tv


    # Calculate alignment between individual tasks
    rank_threshold = 0.97
    print(f"Considering {rank_threshold=}")
    print(f"{datasets=}")

    task_vectors = list(task_vectors_dict.values())

    with torch.no_grad():
        
        # initialize pairwise storage
        allignment_ratios_pairs = {src: {tgt: [] for tgt in datasets} for src in datasets}
    
        # compute subspace overlap. First iterate over all keys
        for current_idx, (current_ds, current_tv) in tqdm(enumerate(task_vectors_dict.items()), 
                                                          total=len(datasets), 
                                                          desc="Computing average subspace ratio..."): 
            
    
            for key in task_vectors[0].vector.keys():
                    
                # Get current weights 
                current_weights = current_tv.vector[key].to(args.device)

                if (len(current_weights.shape) == 2 and key.startswith("model.visual")) and 'text_projection' not in key:
                    # print(f"Processing {key}")
                    U, S, V = torch.linalg.svd(current_weights, full_matrices=False)
                    current_rel_rank = calc_rank(S.cpu(), norm_thresh=rank_threshold)
                    U_current_k = U[:, :current_rel_rank]

                    # Iterate over following task vectors 
                    for next_idx in range(0, len(task_vectors)):
                        
                        next_weights = task_vectors[next_idx].vector[key].to(args.device)
                        
                        proj_next_onto_current_k = torch.linalg.multi_dot((U_current_k, U_current_k.T, next_weights))
                        S_next_k = torch.linalg.svdvals(proj_next_onto_current_k)

                        S_next = torch.linalg.svdvals(next_weights)
                        ar_sum = alignment_ratio(S_next.cpu(), S_next_k.cpu())

                        # allignment_ratios_pairs[current_ds][datasets[next_idx]].append(ar_sum)
                        allignment_ratios_pairs[datasets[next_idx]][current_ds].append(ar_sum)
                else:
                    continue 
                    # print(f"Skipping {key}")


    DATASET_ORDER = [ 'MNIST', 'SVHN', 'GTSRB', 'EuroSAT', 'RESISC45', 'DTD', 'Cars', 'SUN397']
    # compute matrix
    mat = compute_pairwise_avg_matrix(allignment_ratios_pairs, DATASET_ORDER)
    
    fig = plot_pairwise_alignment_matrix(mat, DATASET_ORDER, title="Pair-wise mean SAR", 
                                         save_path=f"./analysis/figs/{args.model}_pairwise_alignment_8ds.pdf",
                                         vmin=0.55,
                                         vmax=1.0,
                                         )
    print("Done!")

    
