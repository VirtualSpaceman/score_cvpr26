import numpy as np
import torch
import os
import math 
import copy 
from src.args import parse_arguments
from src.datasets.registry import registry
from src.constants import get_dict_dataset_paths, get_dict_epochs
from src.merging.task_vectors import TaskVector
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Config
args = parse_arguments()
pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'



def isoc_proj(task_vectors, args):

    device = args.device

    dict_layers = dict()
    
    sv_reduction = 1/len(task_vectors)

    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:

            
            if len(task_vectors[0].vector[key].shape) == 1 or 'model.visual.' not in key: 
                continue 
            
            
            tvs = [task_vector.vector[key].to(device) for task_vector in task_vectors]

            new_vector[key] = sum(tvs) / len(tvs)
            # if len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key:
            # Take only attentions blocks
            if len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key and 'resblock' in key:
                dict_layers[key] = []
                new_vector[key] *= len(tvs)
                    
                
                for tv_idx, tv in enumerate(task_vectors):
                    weights = tv.vector[key].to(device)
                    
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

                    # compute procrustes 
                    u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                    u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                    U_eval = u_u @ v_u
                    V_eval = u_v @ v_v


                # iterate over all weight matrices
                for weight_matrix in tvs: 
                    # Compute the svd decomp. of the current matrix 
                    # U_curr, S_curr, V_curr = torch.linalg.svd(weight_matrix, full_matrices=False)
                    proj_curr = torch.linalg.multi_dot( (U_eval.T, weight_matrix, V_eval.T))

                    dict_layers[key].append(proj_curr.cpu().numpy())

                    
    return dict_layers



# def plot_dict_layers_grid(dict_layers,
#                           out_pdf='dict_layers.pdf',
#                           dpi=150,
#                           global_scale=True):
#     """
#     Create a PDF with a single page showing a grid:
#       - rows: index i (dict_layers[key][i])
#       - columns: keys in dict_layers (layer names)
#     dict_layers: dict -> values are lists (or sequences) of 2D square arrays (numpy arrays)
#     out_pdf: filename to save the PDF to
#     dpi: figure DPI
#     global_scale: if True, use one vmin/vmax across the whole grid;
#                   if False, compute vmin/vmax per layer (per key / per column).
#     """
#     # Ensure consistent ordering
#     keys = list(dict_layers.keys())
#     if len(keys) == 0:
#         raise ValueError("dict_layers is empty")

#     # number of columns and rows
#     cols = len(keys)
#     rows = max(len(lst) for lst in dict_layers.values())

#     # Validate matrices and collect mats for global stats if needed
#     mats_all = []
#     for key, lst in dict_layers.items():
#         for m in lst:
#             arr = np.asarray(m, dtype=float)
#             if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
#                 raise ValueError(f"All matrices must be 2D and square. Problem at key={key}")
#             mats_all.append(arr)
#     if len(mats_all) == 0:
#         raise ValueError("No matrices found inside dict_layers")

#     # Compute global or per-layer vmin/vmax
#     if global_scale:
#         global_min = min(m.min() for m in mats_all)
#         global_max = max(m.max() for m in mats_all)
#         per_layer_limits = None
#     else:
#         per_layer_limits = {}
#         for key, lst in dict_layers.items():
#             if len(lst) == 0:
#                 per_layer_limits[key] = (None, None)
#             else:
#                 arrs = [np.asarray(m, dtype=float) for m in lst]
#                 per_layer_limits[key] = (min(a.min() for a in arrs),
#                                          max(a.max() for a in arrs))
#         global_min = global_max = None

#     # Create figure with a grid of subplots
#     figsize = (max(4, cols * 3), max(3, rows * 3))
#     fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

#     # Normalize axes array shape for consistent indexing
#     if rows == 1 and cols == 1:
#         axes = np.array([[axes]])
#     elif rows == 1:
#         axes = np.atleast_2d(axes)
#     elif cols == 1:
#         axes = np.atleast_2d(axes).T
#     else:
#         axes = np.asarray(axes)

#     # Keep track of last image for global colorbar or per-column colorbars
#     last_im_global = None
#     last_im_per_col = [None] * cols

#     for i in range(rows):
#         for j, key in enumerate(keys):
#             ax = axes[i, j]
#             # If this key has a matrix at index i, plot it
#             if i < len(dict_layers[key]):
#                 mat = np.asarray(dict_layers[key][i], dtype=float)
#                 # choose vmin/vmax
#                 if global_scale:
#                     vmin, vmax = global_min, global_max
#                 else:
#                     vmin, vmax = per_layer_limits[key]
#                 # show matrix
#                 im = ax.imshow(mat, vmin=vmin, vmax=vmax,
#                                aspect='equal', interpolation='nearest')
#                 # store last im for colorbars
#                 if global_scale:
#                     last_im_global = im
#                 else:
#                     last_im_per_col[j] = im

#                 # remove ticks for clarity
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 # label rows on the left
#                 if j == 0:
#                     ax.set_ylabel(f"row {i}", rotation=0, labelpad=30, va='center')
#                 # put column title on the top row
#                 if i == 0:
#                     ax.set_title(str(key), fontsize=9)
#             else:
#                 # missing data: turn off axis and annotate header if top row
#                 ax.axis('off')
#                 if i == 0:
#                     ax.set_title(str(key), fontsize=9)

#     # Layout and colorbars
#     plt.tight_layout()

#     if global_scale:
#         # place a single colorbar for the entire grid (if we plotted at least one image)
#         if last_im_global is not None:
#             # create a single colorbar on the right
#             cax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
#             fig.colorbar(last_im_global, cax=cax)
#             # shrink layout to make sure nothing overlaps
#             plt.subplots_adjust(right=0.92)
#     else:
#         # per-column colorbars (one colorbar beside each column that has images)
#         for j in range(cols):
#             im_for_col = last_im_per_col[j]
#             if im_for_col is not None:
#                 # Attach colorbar to the column (all row axes in that column)
#                 col_axes = axes[:, j] if rows > 1 else np.array([axes[0, j]])
#                 # matplotlib can accept a list of axes to anchor the colorbar
#                 fig.colorbar(im_for_col, ax=col_axes, fraction=0.045, pad=0.02)

#     # Save to a PDF (single page)
#     with PdfPages(out_pdf) as pdf:
#         pdf.savefig(fig, bbox_inches='tight')
#     plt.close(fig)
#     print(f"Saved grid PDF to: {out_pdf}")

def plot_dict_layers_grid_per_matrix(dict_layers,
                                     out_pdf='dict_layers_per_matrix.pdf',
                                     dpi=150,
                                     show_colorbar=True):
    """
    Create a PDF with a single page showing a grid:
      - rows: index i (dict_layers[key][i])
      - columns: keys in dict_layers (layer names)
    Each matrix (each cell) uses its OWN color scale (vmin/vmax computed from that matrix).
    If show_colorbar=True, a small colorbar is drawn next to each plotted matrix.

    Parameters
    ----------
    dict_layers : dict
        values are lists (or sequences) of 2D square numpy arrays
    out_pdf : str
        path to save the single-page PDF
    dpi : int
        figure DPI
    show_colorbar : bool
        if True, draw a small colorbar for each plotted matrix (can be cluttered for large grids)
    """
    # order keys deterministically
    keys = list(dict_layers.keys())
    if len(keys) == 0:
        raise ValueError("dict_layers is empty")

    cols = len(keys)
    rows = max(len(lst) for lst in dict_layers.values())

    # Validate matrices
    total_mats = 0
    for key, lst in dict_layers.items():
        for m in lst:
            arr = np.asarray(m, dtype=float)
            
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(f"All matrices must be 2D and square. Problem at key={key}")
            total_mats += 1
    if total_mats == 0:
        raise ValueError("No matrices found inside dict_layers")

    # Create figure and axes
    figsize = (max(4, cols * 3), max(3, rows * 3))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

    # Normalize axes array shape for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.atleast_2d(axes)
    elif cols == 1:
        axes = np.atleast_2d(axes).T
    else:
        axes = np.asarray(axes)

    # Plot each cell with its own vmin/vmax
    for i in range(rows):
        for j, key in enumerate(keys):
            if 'conv' in key:
                continue 

            ax = axes[i, j]
            if i < len(dict_layers[key]):
                mat = np.asarray(dict_layers[key][i], dtype=float)
                # min max normalization 
                # mat = (mat - mat.min())/(mat.max() - mat.min())
                vmin = float(np.nanmin(mat))
                vmax = float(np.nanmax(mat))
                if vmin == vmax:
                    eps = abs(vmin) * 1e-6 if vmin != 0 else 1e-6
                    vmax = vmin + eps

                im = ax.imshow(mat, vmin=vmin, vmax=vmax,
                               aspect='equal', 
                               interpolation='nearest',
                            #    cmap='RdBu_r',
                               cmap='viridis',
                               )
                ax.set_xticks([])
                ax.set_yticks([])
                # label rows on the left
                if j == 0:
                    ax.set_ylabel(f"row {i}", rotation=0, labelpad=30, va='center')
                # column title on top row
                if i == 0:
                    title_key = str(key).replace('model.visual.','')
                    ax.set_title(title_key, fontsize=9)

                # per-matrix colorbar
                if show_colorbar:
                    # create a small colorbar next to this axis
                    # fig.colorbar accepts ax parameter (a single Axes)
                    # fraction controls size of colorbar; pad controls spacing
                    try:
                        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=6)
                    except Exception:
                        # fallback: if colorbar placement fails, ignore and continue
                        pass
            else:
                # missing data for this row/key
                ax.axis('off')
                if i == 0:
                    title_key = str(key).replace('model.visual.','')
                    ax.set_title(title_key, fontsize=9)

    plt.tight_layout()

    # Save to PDF
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved grid PDF to: {out_pdf}")


if __name__ == '__main__':    
    PATHS = get_dict_dataset_paths()
    EPOCHS = get_dict_epochs()
    task_vectors = []
    dataset_class = registry[args.dataset]
    method = 'seq-ft' if args.sequential_finetuning else 'ind-ft'
    args.data_location = PATHS[args.dataset]
    args.epochs = EPOCHS[args.dataset]

    all_domains = dataset_class.default_domain_order
    domain_names = [dataset_class.BASE_CLASS.DOMAINS[idx] for idx in all_domains]

    for task_idx, domain_idx in enumerate(all_domains):
        args.subset_config = {
            'domains': [dataset_class.BASE_CLASS.DOMAINS[domain_idx]],
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
        
        tv = TaskVector(pretrained_checkpoint, ft_path)
        task_vectors.append(tv)


    save_folder = os.path.join('.', 'analysis', 'figs', args.model, args.dataset)
    os.makedirs(save_folder, exist_ok=True)


    # leave one out 
    # for target_idx, target_domain in enumerate(all_domains): 
    #     # leave the target domain out and marge the remaining 
    #     domain_str = dataset_class.BASE_CLASS.DOMAINS[target_domain]

    #     print(f" [{args.dataset}]Processing plot for subdomain: {domain_str}...")

        
    #     tvs_to_merge = [tv for (idx, tv) in enumerate(task_vectors) if idx != target_idx]

    #     # old version 
    #     dict_layers = isoc_proj(copy.deepcopy(tvs_to_merge), 
    #                             args, )
        
    #     subset_names = [name for name in domain_names if name != domain_str]

    #     for method in ['tsv']:
    #         args.subspace_method = method
    
    #         print(f"Target domain ={domain_str} / {subset_names=} / {method=}")
    #         # new version 
    #         pdf_name = f'spectrum_target_{args.dataset}_{domain_str}_method_{method}.pdf'

    #         # plot_dict_layers_grid(dict_layers=dict_layers,
    #         #                       out_pdf = os.path.join(save_folder, pdf_name),
    #         #                       dpi=250,
    #         #                       global_scale=False)


    #         plot_dict_layers_grid_per_matrix(dict_layers=dict_layers,
    #                         out_pdf = os.path.join(save_folder, pdf_name),
    #                         dpi=250,
    #                         show_colorbar=False)

    #         # plot_dict_layers_grid_per_matrix(dict_layers=dict_layers,
    #         #         out_pdf = os.path.join(save_folder, pdf_name),
    #         #         add_colorbars=False,
    #         #         verbose=True, 
    #         #         global_scale_per_column=True,
    #         #         cmap='YlGnBu_r')



    # include all checkpooints 
    
    domain_str = 'all'

    print(f" [{args.dataset}]Processing plot for subdomain: {domain_str}...")

    # old version 
    dict_layers = isoc_proj(copy.deepcopy(task_vectors), args, )
    
    subset_names = domain_names

    for method in ['tsv']:
        args.subspace_method = method

        print(f"Target domain ={domain_str} / {subset_names=} / {method=}")
        # new version 
        pdf_name = f'results_layerwise_proj_target_{args.dataset}_{domain_str}_method_{method}.pdf'

        # plot_dict_layers_grid(dict_layers=dict_layers,
        #                       out_pdf = os.path.join(save_folder, pdf_name),
        #                       dpi=250,
        #                       global_scale=False)


        plot_dict_layers_grid_per_matrix(dict_layers=dict_layers,
                        out_pdf = os.path.join(save_folder, pdf_name),
                        dpi=250,
                        show_colorbar=False)

        # plot_dict_layers_grid_per_matrix(dict_layers=dict_layers,
        #         out_pdf = os.path.join(save_folder, pdf_name),
        #         add_colorbars=False,
        #         verbose=True, 
        #         global_scale_per_column=True,
        #         cmap='YlGnBu_r')
    
    print("Done.")