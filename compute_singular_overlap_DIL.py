import numpy as np
import torch
from pprint import pprint
import json 
import os
import copy 
from src.merging.task_vectors import TaskVector
from src.args import parse_arguments
from src.datasets.registry import registry
from src.constants import get_dict_dataset_paths, get_dict_epochs
from tqdm import tqdm 
import math

import matplotlib.pyplot as plt

from src.merging.tsvm import tsv_ood

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
            ax.text(j, i, txt, ha='center', va='center', color=text_color, fontsize=16)

    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(rotation=45, ha='right', fontsize=FONT_SIZE)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        print(f"Saved heatmap to: {save_path}")

    return fig


def get_merged_dict(task_vectors, subspace_method, args):
    new_vector = dict()

    sv_reduction = 1/len(task_vectors)
    
    for key in task_vectors[0].vector.keys():

        if len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key:
            
            # Concatenate all decompositions for each layer 
            for tv_idx, tv in enumerate(task_vectors):
            
                weights = tv.vector[key].to(args.device)
                
                u, s, v = torch.linalg.svd(weights, full_matrices=False)

                if tv_idx == 0:
                    # print(f"Computed SVD for {key}...")
                    sum_u = torch.zeros_like(u, device=args.device)
                    sum_s = torch.zeros_like(s, device=args.device)
                    sum_v = torch.zeros_like(v, device=args.device)
                reduced_index_s = int(s.shape[0] * sv_reduction)

                # select only the first reduced_index_s columns of u and place them
                sum_u[:, tv_idx * reduced_index_s : (tv_idx + 1) * reduced_index_s] = u[
                    :, :reduced_index_s
                ]
                sum_s[tv_idx * reduced_index_s : (tv_idx + 1) * reduced_index_s] = s[
                    :reduced_index_s
                ]
                # select only the first reduced_index_s rows of v and place them
                sum_v[tv_idx * reduced_index_s : (tv_idx + 1) * reduced_index_s, :] = v[
                    :reduced_index_s, :
                ]

            # Based on the concatenated version, now choose which U and V we are using 
            if subspace_method == 'tsv':
                # process 
                u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                U_new = u_u @ v_u
                V_new = u_v @ v_v
            else:
                U_new = sum_u
                V_new = sum_v

            
            # define the sigma variable 
            new_sigma = 0
            if subspace_method == 'tsv':
                for _, task_vector in enumerate(task_vectors):
                    current_change_basis = U_new.T @ task_vector.vector[key].to(args.device) @ V_new.T
                    
                    new_sigma += torch.diag(current_change_basis)

                # Transform from 1d array to 2d diagonal matrix 
                new_sigma = torch.diag(new_sigma)
            else:
                for _, task_vector in enumerate(task_vectors):
                    current_change_basis = U_new.T @ task_vector.vector[key].to(args.device) @ V_new.T

                    #  Zero out diagonal elements
                    off_diag = current_change_basis.clone()
                    off_diag.diagonal(dim1=-1, dim2=-2).zero_()

                    # Apply thresholding to off-diagonal elements
                    off_diag_mean = off_diag.mean()
                    off_diag_std = off_diag.std()
                    threshold = 1.96 * off_diag_std #-> bom para o caso medico
                    off_diag_mask = (off_diag - off_diag_mean).abs() <= threshold
                    # print(f"Count: {off_diag_mask.sum()/off_diag.numel()}")
                    off_diag *= off_diag_mask

                    # Extract diagonal and update new_sigma
                    diag_only = torch.diag_embed(torch.diagonal(current_change_basis, dim1=-2, dim2=-1))
                    new_sigma += diag_only + off_diag
                

                # use the full matrix and take the average 
                new_sigma /= len(task_vectors)
                

            new_vector[key] = torch.linalg.multi_dot(
                (
                    U_new,
                    new_sigma,
                    V_new,
                )
            )

    return new_vector


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
    task_vectors = []
    dataset_class = registry[args.dataset]
    method = 'seq-ft' if args.sequential_finetuning else 'ind-ft'
    args.data_location = PATHS[args.dataset]
    args.epochs = EPOCHS[args.dataset]

    all_domains = dataset_class.default_domain_order
    DOMAINS_NAMES = [dataset_class.BASE_CLASS.DOMAINS[idx] for idx in all_domains]
    

    task_vectors_dict = dict()
    for task_idx, domain_idx in enumerate(all_domains):
        domain_str = dataset_class.BASE_CLASS.DOMAINS[domain_idx]
        args.subset_config = {
            'domains': [domain_str],
            'classes': dataset_class.BASE_CLASS.CLASSES,
        }
        subset_config_id = dataset_class.BASE_CLASS.get_md5(args.subset_config)
        if args.sequential_finetuning:
            args.save = f'checkpoints/{args.model}/sequential_finetuning/domain_incremental'
            ckpdir = os.path.join(args.save, args.dataset)
            ft_path = os.path.join(ckpdir, f'checkpoint_ep:{args.epochs}-lr:{args.lr}_{task_idx}.pt')
        else:
            args.save = f'checkpoints/{args.model}/domain_incremental'
            ckpdir = os.path.join(args.save, args.dataset)
            ft_path = os.path.join(ckpdir, f'checkpoint_ep:{args.epochs}-lr:{args.lr}_{subset_config_id}.pt')
        
        task_vectors_dict[domain_str] = TaskVector(pretrained_checkpoint, ft_path)


    task_vectors = list(task_vectors_dict.values())

    # Calculate alignment between individual tasks
    rank_threshold = 0.97

    print(f"Considering {rank_threshold=}")
    print(f"{DOMAINS_NAMES=}")

    
    with torch.no_grad():
        
        # initialize pairwise storage
        allignment_ratios_pairs = {src: {tgt: [] for tgt in DOMAINS_NAMES} for src in DOMAINS_NAMES}
    
        # compute subspace overlap. First iterate over all keys
        for current_idx, (current_ds, current_tv) in tqdm(enumerate(task_vectors_dict.items()), 
                                                          total=len(DOMAINS_NAMES), 
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
                        # U_next_k, S_next_k, V_next_k = torch.linalg.svd(proj_next_onto_current_k, full_matrices=False)
                        S_next_k = torch.linalg.svdvals(proj_next_onto_current_k)
                        
                        # U_next, S_next, V_next = torch.linalg.svd(next_weights, full_matrices=False)
                        S_next = torch.linalg.svdvals(next_weights)
                        ar_sum = alignment_ratio(S_next.cpu(), S_next_k.cpu())

                        # allignment_ratios_pairs[current_ds][DOMAINS_NAMES[next_idx]].append(ar_sum)
                        allignment_ratios_pairs[DOMAINS_NAMES[next_idx]][current_ds].append(ar_sum)

                else:
                    continue 
                    # print(f"Skipping {key}")

    # avg_alignment_ratios_sum = {ds: np.round(np.mean(ar), 3).item() for ds, ar in allignment_ratios_sum.items() if len(ar) > 0}
    # pprint(f"Average alignment ratios for {current_ds}: {avg_alignment_ratios_sum}")

    # compute matrix
    mat = compute_pairwise_avg_matrix(allignment_ratios_pairs, DOMAINS_NAMES)
    
    fig = plot_pairwise_alignment_matrix(mat, DOMAINS_NAMES, title="Pair-wise mean SAR", 
                                         save_path=f"./analysis/figs/{args.model}_pairwise_alignment_domain_{args.dataset}.pdf",
                                         vmin=0.55, 
                                         vmax=1.0,
                                         )
    print("Done!")






        



