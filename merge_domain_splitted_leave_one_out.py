import numpy as np
import torch
import json 
import os
import copy 
from pprint import pprint
from src.merging.task_vectors import TaskVector
from src.merging.ties import vector_to_state_dict
from src.eval import eval_given_dataset, get_dataset
from src.args import parse_arguments
from src.datasets.registry import registry
from src.constants import get_dict_dataset_paths, get_dict_epochs
from src.merging.utils import get_merged_tv, compute_same_sign_score
from src.merging.pcb import get_pcb_inference_weights

# Config
args = parse_arguments()
pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'




def get_inter_range(merge_fn: str):
    if merge_fn == 'pcb':
        aug_range = np.linspace(0.8, 2.5, endpoint=True, num=18) # nlp / mtl 
    elif merge_fn in ['ties', 'dare_ties']: 
        aug_range = np.linspace(0.8, 1.8, endpoint=True, num=11) # Ties original -> NLP / MTL 
    elif merge_fn in ['tsv']:   
        aug_range = np.linspace(0.6, 1.5, endpoint=True, num=10) # paper fig 4
    elif merge_fn in ['isoc']:
        aug_range = np.linspace(0.1, 1.5, endpoint=True, num=15)
    elif merge_fn in ['saliency']:
        aug_range = np.linspace(0.1, 1.5, endpoint=True, num=15)
    elif merge_fn in ['saliency_spectrum']:
        aug_range = np.linspace(0.1, 1.5, endpoint=True, num=15)
    elif merge_fn in ['saliency_precomp', 'randiso', 
                      'saliency_ties', 'isoc_proj_es'] or merge_fn.startswith('isoc_changeb'):
        aug_range = [1.0]
    else:
        aug_range = np.linspace(0.1, 1.0, endpoint=True, num=10) 

    print(f"[{merge_fn=}] - Coefficient range: {aug_range}")
    return aug_range



if __name__ == '__main__':    
    PATHS = get_dict_dataset_paths()
    EPOCHS = get_dict_epochs()
    task_vectors = []
    dataset_class = registry[args.dataset]
    method = 'seq-ft' if args.sequential_finetuning else 'ind-ft'
    args.data_location = PATHS[args.dataset]
    args.epochs = EPOCHS[args.dataset]

    all_domains = dataset_class.default_domain_order

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


    # setup dict to log results 
    results_dict = dict()
    
    if not args.debug: 
        save_path = os.path.join("./results", "merging", args.model, args.dataset, args.merge_fn, method)
        os.makedirs(save_path, exist_ok=True)

    first_key = f"{args.merge_fn}_{args.dataset}"
    results_dict[first_key] = dict()
    # merge_fn -> dataset -> domains -> -> acc

    # Get scaling factors 
    # scaling_factors = get_inter_range(args.merge_fn)
    scaling_factors = [1.0]
    if args.debug:
        scaling_factors = [1.0]


    for target_idx, target_domain in enumerate(all_domains): 
        # leave the target domain out and marge the remaining 
        domain_str = dataset_class.BASE_CLASS.DOMAINS[target_domain]
        results_dict[first_key][domain_str] = dict()

        args.subset_config = {
            'domains': [domain_str],
            'classes': dataset_class.BASE_CLASS.CLASSES,
        }
        tvs_to_merge = [tv for (idx, tv) in enumerate(task_vectors) if idx != target_idx]
 
        # precompute alpha if need
        # if args.compute_alpha:
        if args.merge_fn in ['saliency_precomp',
                             'saliency_ties',
                             ] or args.merge_fn.startswith('ties_sigma'): 
            # print(f"precomputing alpha based on task vector similarites...")
            # cos_sim = compute_sim(copy.deepcopy(tvs_to_merge)) # returns a np.array 
            # # set main diag to zero because it is always 1.0
            # cos_sim -= np.eye(len(tvs_to_merge))
            # ref_value = cos_sim.max()

            print(f"precomputing alpha based on sign agreements...")
            ref_value = compute_same_sign_score(copy.deepcopy(tvs_to_merge))

            # then take the maximum value 
            # MULT = 1.5 # 1.5 eh ok for the rest 
            MULT = 1.0 # 1.0 for spec tries  
            selected_alpha = np.round(MULT*ref_value, 3) # -> usava ,2 
            selected_alpha = max(selected_alpha, 0.03)

            print(f"Setting alpha to {selected_alpha=} -> {MULT}*Ref value: ")
            args.alpha = selected_alpha


        merged_tv = get_merged_tv(copy.deepcopy(tvs_to_merge), 
                                      args)

        # if args.debug:
        #     exit(0)
        # Iterate over all scaling factors
        for scaling_factor in scaling_factors:


            scaling_factor = np.round(scaling_factor, 2)
            # Adjust the vector according the scaling factor 
            if args.merge_fn == 'pcb':
                clamp_tvs, scale = merged_tv
                print("No augment! ")
                pcb_tv = get_pcb_inference_weights(clamp_tvs, scale, scaling_factor)
                pcb_tv = TaskVector(vector=vector_to_state_dict(pcb_tv, state_dict=task_vectors[0].vector))
                eval_tv = pcb_tv
            else: 
                eval_tv = merged_tv * scaling_factor

            # Create image encoder based on the task vector 
            image_encoder = eval_tv.apply_to(pretrained_checkpoint, scaling_coef=1.0)

            preprocess_fn = image_encoder.val_preprocess
            target_dataset = get_dataset(
                            args.dataset,
                            preprocess_fn,
                            location=args.data_location,
                            batch_size=args.batch_size,
                            subset_config=args.subset_config,
                        )
            r = eval_given_dataset(image_encoder, target_dataset, args.dataset, args)
            results_dict[first_key][domain_str][f"acc_{scaling_factor}"] = r

        
        # if args.debug:
        #     break 
    
    # flat_list = list(results_dict.values())
    # avg_top1_results = np.round(np.mean([x['top1'] for x in flat_list]).item(), 3).item() * 100
    # avg_balacc_results = np.round(np.mean([x['bal_acc'] for x in flat_list]).item(), 3).item() * 100
    # print(f"[Scaling={scaling_factor}] Avg. Top1: {avg_top1_results}%...")
    # print(f"[Scaling={scaling_factor}] Avg. Bal.Acc: {avg_balacc_results}%...")
    # pprint(results_dict)

    
    avg_top1_results = np.round(np.mean([results_dict[first_key][domain]["acc_1.0"]['top1'] for domain in results_dict[first_key]]).item(), 3).item() * 100
    avg_balacc_results = np.round(np.mean([results_dict[first_key][domain]["acc_1.0"]['bal_acc'] for domain in results_dict[first_key]]).item(), 3).item() * 100
    print(f"[Scaling={scaling_factor}] Avg. Bal.Acc: {avg_balacc_results}%...")
    print(f"[Scaling={scaling_factor}] Avg. Top1: {avg_top1_results}%...")

    if not args.debug: 
        name=f"merging-{args.model}-{args.dataset}-DIL-{method}"
        json_filename = f"{name}.json"
        json_path = os.path.join(save_path, json_filename)
        with open(json_path, 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)