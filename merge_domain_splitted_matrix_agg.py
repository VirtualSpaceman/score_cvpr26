import numpy as np
import torch
import os
import math 
import copy 
from src.args import parse_arguments
from src.datasets.registry import registry
from src.constants import get_dict_dataset_paths, get_dict_epochs
from src.merging.task_vectors import TaskVector

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Config
args = parse_arguments()
pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'


# Define the color scale values and corresponding colors
# colors = [
#     ( 0.0, '#b2182b'),  # deep red
#     ( 0.1, '#d6604d'),
#     # ( 0.3, '#f4a582'),
#     # ( 0.4, '#fddbc7'),
#     # ( 0.4, '#f7f7f7'),
#     ( 0.2, '#ffffff'),  # white midpoint
#     ( 0.5, '#f7f7f7'),
#     ( 0.6, '#d1e5f0'),
#     ( 0.8, '#92c5de'),
#     ( 0.9, '#4393c3'),
#     ( 1.0, '#2166ac')   # deep blue
# ]
colors = [
    # ( 0.1, '#d6604d'),
    # ( 0.1, '#4393c3'),
    ( 0.05, '#2166ac'),   # deep blue
    ( 0.10, '#ffffff'),  # white midpoint
    # ( 0.10, '#92c5de'),
    # ( 0.10, '#f4a582'),
    ( 0.15, '#fddbc7'),
    # ( 0.2, '#f7f7f7'),
    # ( 0.3, '#f7f7f7'),
    # ( 0.25, '#d1e5f0'),
    ( 0.18, '#b2182b'),  # deep red
]

# Extract just the colors (ignoring the numeric positions)
color_list = [c for _, c in colors]

# Create the continuous colormap
custom_cmap = LinearSegmentedColormap.from_list("custom_RdBu_like", color_list, N=6)

# (Optional) register it globally for reuse
plt.colormaps.register(name="custom_RdBu_like", cmap=custom_cmap)


def isoc_proj(task_vectors, args):

    device = args.device

    dict_layers = dict()
    
    sv_reduction = 1/len(task_vectors)

    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:

            
            if len(task_vectors[0].vector[key].shape) == 1 or 'model.visual.' not in key: 
                continue 
            
            dict_layers[key] = []
            tvs = [task_vector.vector[key].to(device) for task_vector in task_vectors]

            new_vector[key] = sum(tvs) / len(tvs)

            if len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key:
                new_vector[key] *= len(tvs)

                if args.subspace_method in ['tsv']:
                    
                    sv_reduction = 1/len(task_vectors)
                    
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


                    # process 
                    if args.subspace_method == 'tsv':
                        u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                        u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                        U_eval = u_u @ v_u
                        V_eval = u_v @ v_v

                new_diag = 0
                # iterate over all weight matrices
                for weight_matrix in tvs: 
                    # Compute the svd decomp. of the current matrix 
                    # U_curr, S_curr, V_curr = torch.linalg.svd(weight_matrix, full_matrices=False)
                    proj_curr = torch.linalg.multi_dot( (U_eval.T, weight_matrix, V_eval.T))

                    # if args.subspace_method == 'concat':
                    #     #  Zero out diagonal elements
                    #     off_diag = proj_curr.clone()
                    #     off_diag.diagonal(dim1=-1, dim2=-2).zero_()

                    #     # Apply thresholding to off-diagonal elements
                    #     off_diag_mean = off_diag.mean()
                    #     off_diag_std = off_diag.std()
                    #     threshold = 1.96 * off_diag_std #-> bom para o caso medico
                    #     off_diag_mask = (off_diag - off_diag_mean).abs() <= threshold
                    #     # print(f"Count: {off_diag_mask.sum()/off_diag.numel()}")
                    #     off_diag *= off_diag_mask

                    #     # Extract diagonal and update new_sigma
                    #     diag_only = torch.diag_embed(torch.diagonal(proj_curr, dim1=-2, dim2=-1))
                    #     new_diag += diag_only + off_diag
                    # else: 
                    #     new_diag += torch.diag(proj_curr)

                    new_diag += proj_curr
                
                # new_diag /= len(tvs) 
                # U_p, S_p, Vh_p = torch.linalg.svd(new_diag, full_matrices=False)


                # Reconstruct
                # new_diag = U_p @ Vh_p
                # new_diag = U_p @ torch.diag(S_p) @ Vh_p
                
                # if args.subspace_method == 'avg':
                #     new_diag /= len(task_vectors)
                #     new_diag = torch.diag(new_diag)
                # elif args.subspace_method == 'tsv':
                #     new_diag = torch.diag(new_diag)
                # elif args.subspace_method == 'concat':
                #     new_diag /= len(task_vectors)
                # else:
                #     raise ValueError("Invalid method")

                # assure that is a 2d matrix
                assert len(new_diag.shape) == 2
                dict_layers[key] = new_diag.cpu().numpy()
                # print(f"{key=} / {dict_layers[key].shape}" )
                    
    return dict_layers



# def plot_dict_layers_grid_per_matrix(dict_layers,
#                                      out_pdf='dict_layers_per_matrix.pdf',
#                                      dpi=150,
#                                      show_colorbar=True):
#     """
#     Create a PDF with a single page showing a grid:
#       - rows: index i (dict_layers[key][i])
#       - columns: keys in dict_layers (layer names)
#     Each matrix (each cell) uses its OWN color scale (vmin/vmax computed from that matrix).
#     If show_colorbar=True, a small colorbar is drawn next to each plotted matrix.

#     Parameters
#     ----------
#     dict_layers : dict
#         values are lists (or sequences) of 2D square numpy arrays
#     out_pdf : str
#         path to save the single-page PDF
#     dpi : int
#         figure DPI
#     show_colorbar : bool
#         if True, draw a small colorbar for each plotted matrix (can be cluttered for large grids)
#     """
#     # order keys deterministically
#     keys = list(dict_layers.keys())
#     if len(keys) == 0:
#         raise ValueError("dict_layers is empty")

#     cols = len(keys)
#     rows = max(len(lst) for lst in dict_layers.values())

#     # Validate matrices
#     total_mats = 0
#     for key, lst in dict_layers.items():
#         for m in lst:
#             arr = np.asarray(m, dtype=float)
            
#             if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
#                 raise ValueError(f"All matrices must be 2D and square. Problem at key={key}")
#             total_mats += 1
#     if total_mats == 0:
#         raise ValueError("No matrices found inside dict_layers")

#     # Create figure and axes
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

#     # Plot each cell with its own vmin/vmax

#     for j, key in enumerate(keys):
#         if 'conv' in key:
#             continue 

#         ax = axes[i, j]
#         if i < len(dict_layers[key]):
#             mat = np.asarray(dict_layers[key][i], dtype=float)
#             # min max normalization 
#             mat = (mat - mat.min())/(mat.max() - mat.min())
#             vmin = float(np.nanmin(mat))
#             vmax = float(np.nanmax(mat))
#             if vmin == vmax:
#                 eps = abs(vmin) * 1e-6 if vmin != 0 else 1e-6
#                 vmax = vmin + eps

#             im = ax.imshow(mat, vmin=vmin, vmax=vmax,
#                             aspect='equal', 
#                             interpolation='nearest',
#                             cmap='RdBu_r')
#             ax.set_xticks([])
#             ax.set_yticks([])
#             # label rows on the left
#             if j == 0:
#                 ax.set_ylabel(f"row {i}", rotation=0, labelpad=30, va='center')
#             # column title on top row
#             if i == 0:
#                 title_key = str(key).replace('model.visual.','')
#                 ax.set_title(title_key, fontsize=9)

#             # per-matrix colorbar
#             if show_colorbar:
#                 # create a small colorbar next to this axis
#                 # fig.colorbar accepts ax parameter (a single Axes)
#                 # fraction controls size of colorbar; pad controls spacing
#                 try:
#                     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#                     cbar.ax.tick_params(labelsize=6)
#                 except Exception:
#                     # fallback: if colorbar placement fails, ignore and continue
#                     pass
#         else:
#             # missing data for this row/key
#             ax.axis('off')
#             if i == 0:
#                 title_key = str(key).replace('model.visual.','')
#                 ax.set_title(title_key, fontsize=9)

#     plt.tight_layout()

#     # Save to PDF
#     with PdfPages(out_pdf) as pdf:
#         pdf.savefig(fig, bbox_inches='tight')
#     plt.close(fig)
#     print(f"Saved grid PDF to: {out_pdf}")


def plot_dict_layers_grid_per_matrix(dict_layers,
                                     out_pdf='dict_layers_per_matrix.pdf',
                                     dpi=150,
                                     args = None, 
                                     show_colorbar=True):
    """
    Create a PDF with a single page showing a "square mosaic" grid of all matrices.
    The grid dimensions (rows, cols) are calculated to be as square as possible
    to contain all matrices.
    
    Each matrix (each cell) uses its OWN color scale (vmin/vmax computed from that matrix
    after min-max normalization).

    Parameters
    ----------
    dict_layers : dict
        A dictionary where keys are layer_name (str) and values are lists 
        (or sequences) of 2D square numpy arrays. 
        Example: {'layer1': [mat1, mat2], 'layer2': [mat3]}
    out_pdf : str
        path to save the single-page PDF
    dpi : int
        figure DPI
    show_colorbar : bool
        if True, draw a small colorbar for each plotted matrix.
    """
    
    # --- 1. Flatten data and filter ---
    # Create a flat list of all matrices to plot, along with their metadata.
    # This also filters out 'conv' layers.
    plot_items = []
    for key, value in dict_layers.items():
        if 'conv' in key:
            continue

        arr = np.array(value, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            print(key, arr.shape)
            raise ValueError(
                f"All matrices must be 2D and square. Problem at key={key}, index={i}"
            )
        plot_items.append({'key': key, 'matrix': arr})

    total_mats = len(plot_items)
    if total_mats == 0:
        print("No valid matrices found to plot (after filtering 'conv' layers).")
        return

    # --- 2. Calculate "Square Mosaic" Grid Dimensions ---
    # This is the new logic for a square-ish layout
    cols = int(math.ceil(math.sqrt(total_mats)))
    rows = int(math.ceil(total_mats / cols))

    # --- 3. Create Figure and Axes ---
    figsize = (max(4, cols * 3.5), max(3, rows * 3))
    fig, axes = plt.subplots(rows, cols, 
                            figsize=figsize, dpi=dpi)

    fig.suptitle(f"[{args.subspace_method}] Dataset = {args.dataset} -> Target: {args.target_domain}", 
                 fontsize=16)
    
    # Flatten axes array for easy iteration
    if total_mats == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    # --- 4. Plot each matrix in the mosaic grid ---
    for k in range(total_mats):
        item = plot_items[k]
        mat = item['matrix']
        key = item['key']
        ax = axes_flat[k]

        # Min-max normalization for this specific matrix
        mat_norm = mat
        if mat.max() != mat.min():
             mat_norm = (mat - mat.min()) / (mat.max() - mat.min())
        
        vmin = float(np.nanmin(mat_norm))
        vmax = float(np.nanmax(mat_norm))
        
        # Handle cases where matrix is all one value
        if vmin == vmax:
            eps = abs(vmin) * 1e-6 if vmin != 0 else 1e-6
            vmax = vmin + eps

        im = ax.imshow(mat_norm, 
                       vmin=vmin, vmax=vmax,
                        aspect='equal', 
                        interpolation='nearest',
                        # cmap='RdBu_r',
                        cmap='Accent',
                        # cmap=custom_cmap,
                        # cmap='Paired',
                        # cmap='tab10',
                        # cmap='Set2',

                        )
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set a title for each subplot indicating its origin
        title_key = str(key).replace('model.visual.', '') # Shorten key
        ax.set_title(f"{title_key}", fontsize=9)

        # Per-matrix colorbar
        if show_colorbar:
            try:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=6)
            except Exception:
                pass  # fallback: if colorbar fails, continue

    # --- 5. Turn off unused axes ---
    for k in range(total_mats, len(axes_flat)):
        axes_flat[k].axis('off')

    # if args is not None: 
    #     plt.suptitle(,
    #                  x=0.0)
    
    plt.tight_layout()

    # --- 6. Save to PDF ---
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved square mosaic grid PDF to: {out_pdf}")



# def plot_dict_layers_row(dict_layers,
#                         out_pdf='dict_layers_per_matrix.pdf',
#                         dpi=150,
#                         show_colorbar=True):

#     new_suptitles = ['High agreement and minimal off-diagonal conflicts',
#                      'High agreement and moderate off-diagonal conflicts',
#                      'Low agreement and moderate off-diagonal conflicts',
#                      ]
    
#     plt.rcParams.update({
#         # 'font.size': 10,
#         'font.family': 'sans-serif',
#     })
#     # --- 1. Flatten data and filter ---
#     # Create a flat list of all matrices to plot, along with their metadata.
#     # This also filters out 'conv' layers.
#     plot_items = []
#     for key, value in dict_layers.items():

#         arr = np.array(value, dtype=float)
#         if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
#             print(key, arr.shape)
#             raise ValueError(
#                 f"All matrices must be 2D and square. Problem at key={key}, index={i}"
#             )
#         plot_items.append({'key': key, 'matrix': arr})

#     total_mats = len(plot_items)
#     if total_mats == 0:
#         print("No valid matrices found to plot (after filtering 'conv' layers).")
#         return

    
#     cols = total_mats
#     rows = 1
#     # --- 3. Create Figure and Axes ---
#     figsize = (max(4, cols * 3.5), max(4, rows * 3))
#     fig, axes = plt.subplots(rows, cols, 
#                             figsize=figsize, dpi=dpi)

    
#     # Flatten axes array for easy iteration
#     if total_mats == 1:
#         axes_flat = [axes]
#     else:
#         axes_flat = axes.flatten()

#     # --- 4. Plot each matrix in the mosaic grid ---
#     for k in range(total_mats):
#         item = plot_items[k]
#         mat = item['matrix']
#         key = item['key']
#         ax = axes_flat[k]

#         # Min-max normalization for this specific matrix
#         mat_norm = mat
#         if mat.max() != mat.min():
#              mat_norm = (mat - mat.min()) / (mat.max() - mat.min())
        
#         vmin = float(np.nanmin(mat_norm))
#         vmax = float(np.nanmax(mat_norm))
        
#         # Handle cases where matrix is all one value
#         if vmin == vmax:
#             eps = abs(vmin) * 1e-6 if vmin != 0 else 1e-6
#             vmax = vmin + eps

#         im = ax.imshow(mat_norm, 
#                        vmin=vmin, 
#                        vmax=vmax,
#                         aspect='equal', 
#                         interpolation='nearest',
#                         # cmap='RdBu_r',
#                         # cmap='Accent',
#                         cmap=custom_cmap,
#                         # cmap='Paired',
#                         # cmap='tab10',
#                         # cmap='Set2',

#                         )
        
#         title_key = str(key).replace('model.visual.', '') # Shorten key
#         # ax.set_title(f"{title_key}", fontsize=9)
#         if total_mats == len(new_suptitles):
#             ax.set_title(f"{new_suptitles[k]}", 
#                         fontsize=7)
        
#         ax.set_xticks([])
#         ax.set_yticks([])
        

    

#     # Leave a bit of space on the right for the colorbar
#     fig.subplots_adjust(right=0.88, hspace=0.0, wspace=0.05)
#     # Create a dedicated axes for the colorbar
#     cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
#     cbar = fig.colorbar(im, cax=cbar_ax, 
#                         orientation='vertical', 
#                         extend='both', fraction=0.7,
#                         )
#     cbar.ax.tick_params(labelsize=9)


#     # fig.subplots_adjust(hspace=0.0, wspace=0.05)
#     # --- 5. Turn off unused axes ---
#     for k in range(total_mats, len(axes_flat)):
#         axes_flat[k].axis('off')
    

#     # --- 6. Save to PDF ---
#     with PdfPages(out_pdf) as pdf:
#         pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
#     print(f"Saved square mosaic grid PDF to: {out_pdf}")




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



    # leave one out 
    # for target_idx, target_domain in enumerate(all_domains): 
    #     # leave the target domain out and marge the remaining 
    #     domain_str = dataset_class.BASE_CLASS.DOMAINS[target_domain]
    #     args.target_domain = domain_str
    #     print(f" [{args.dataset}]Processing plot for subdomain: {domain_str}...")

        
    #     tvs_to_merge = [tv for (idx, tv) in enumerate(task_vectors) if idx != target_idx]
        
    #     subset_names = [name for name in domain_names if name != domain_str]

    #     # for method in ['avg', 'tsv', 'concat']:
    #     for method in ['tsv']:

    #         save_folder = os.path.join('.', 'analysis', 'figs', args.model, args.dataset, f"comb_strat_{method}")
    #         os.makedirs(save_folder, exist_ok=True)
            
    #         args.subspace_method = method
    
    #         # old version 
    #         dict_layers = isoc_proj(copy.deepcopy(tvs_to_merge), 
    #                                 args, )
        
    #         print(f"Target domain = {domain_str} / {subset_names=} / {method=}")

    #         pdf_name = f'result_spec_target_{args.dataset}_{domain_str}_method_{method}.pdf'

    #         plot_dict_layers_grid_per_matrix(dict_layers=dict_layers,
    #                         out_pdf = os.path.join(save_folder, pdf_name),
    #                         dpi=250,
    #                         args=args,
    #                         show_colorbar=True)

    #         # plot_dict_layers_grid_per_matrix(dict_layers=dict_layers,
    #         #         out_pdf = os.path.join(save_folder, pdf_name),
    #         #         add_colorbars=False,
    #         #         verbose=True, 
    #         #         global_scale_per_column=True,
    #         #         cmap='YlGnBu_r')



    # all 
     
    keep_only_layers = [
                        '.1.attn.out_proj.weight', # diag  
                        # '.1.attn.in_proj_weight' , # diag  
                        # '3.attn.in_proj',
                        # '4' 
                        # '6',
                        '5.attn.out_proj',
                        # '5', 
                        # '5.out_proj.weight',
                        '5.mlp.c_fc.weight',
                        # '8', 
                        # '8.attn.out_proj.weight', ## full noise 
                        # '9.mlp.c_fc.weight', ##full noise
                        # '9',
                        ]

    keep_only_layers = []
    domain_str = 'all'
    args.target_domain = domain_str
    print(f" [{args.dataset}]Processing plot for subdomain: {domain_str}...")
    
    subset_names = domain_names

    # for method in ['avg', 'tsv', 'concat']:
    for method in ['tsv']:

        save_folder = os.path.join('.', 'analysis', 'figs', args.model, args.dataset, f"comb_strat_{method}")
        os.makedirs(save_folder, exist_ok=True)
        
        args.subspace_method = method

        # old version 
        dict_layers = isoc_proj(copy.deepcopy(task_vectors), 
                                args, )

        # Filter 
        if len(keep_only_layers) > 0:
            # new_dict = {key: v for (key, v) in dict_layers.items() if any([x in key for x in keep_only_layers])}
            new_dict = dict()
            for keep in keep_only_layers:
                for key in dict_layers:
                    if keep in key:
                        new_dict[key] = dict_layers[key]
            # print(f"New layer dict: {new_dict}")
            dict_layers = new_dict
             
        print(f"Target domain = {domain_str} / {subset_names=} / {method=}")


        pdf_name = f'result_spec_target_{args.dataset}_{domain_str}_method_{method}.pdf'
        plot_dict_layers_grid_per_matrix(dict_layers=dict_layers,
                        out_pdf = os.path.join(save_folder, pdf_name),
                        dpi=250,
                        args=args,
                        show_colorbar=True,
        )
        
        
        pdf_name = f'row_result_spec_target_{args.dataset}_{domain_str}_method_{method}.pdf'
        # plot_dict_layers_row(dict_layers=dict_layers,
        #         out_pdf = os.path.join(save_folder, pdf_name),
        #         dpi=250,
        #         show_colorbar=True)

    
    print("Done!")