import numpy as np
import torch
from pprint import pprint
import json 
import os
import wandb
import copy 
from src.merging.task_vectors import TaskVector
from src.merging.breadcumbs import TaskVectorKeepTop
from src.merging.ties import vector_to_state_dict
from src.eval import eval_given_dataset, get_dataset
from src.args import parse_arguments
from src.datasets.registry import registry
from src.constants import get_dict_dataset_paths, get_dict_epochs
from src.merging.utils import get_merged_tv
from src.merging.pcb import get_pcb_inference_weights

# Config
args = parse_arguments()
pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'



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
        # tv = TaskVectorKeepTop(pretrained_checkpoint, ft_path, top_k_keep=args.alpha)
        task_vectors.append(tv)


    if not args.debug: 
        project_name = 'merging-DIL-ood'
        wandb.init(
            project=project_name,
            entity=args.wandb_entity_name,
            mode='offline',
            name=f"merging-{args.dataset}-{method}-{args.merge_fn}",
            config=args,
            tags=["merging", "DIL", method],
        )
    
    # Get scaling factors 
    scaling_factors = np.linspace(0.1, 1.5, endpoint=True, num=15)

    if args.debug:
        scaling_factors = [1.0]

    print(f"Considering scaling factors: {scaling_factors=}")

    for scaling_factor in scaling_factors:
        # Iterate over all scaling factors
        scaling_factor = np.round(scaling_factor, 2)

        # setup dict to log results 
        results_dict = dict()
        
        for target_idx, target_domain in enumerate(all_domains): 

            # leave the target domain out and marge the remaining 
            domain_str = dataset_class.BASE_CLASS.DOMAINS[target_domain]

            args.subset_config = {
                'domains': [domain_str],
                'classes': dataset_class.BASE_CLASS.CLASSES,
            }
            tvs_to_merge = [tv for (idx, tv) in enumerate(task_vectors) if idx != target_idx]


            # precompute alpha if need
            # if args.compute_alpha:
            if args.merge_fn in ['saliency_precomp']: 
                print(f"precomputing alpha based on task vector similarites...")
                cos_sim = compute_sim(copy.deepcopy(tvs_to_merge)) # returns a np.array 

                # set main diag to zero because it is always 1.0
                cos_sim -= np.eye(len(tvs_to_merge))
                
                # then take the maximum value 
                MULT = 1.5
                selected_alpha = np.round(MULT*cos_sim.max(), 2)

                print(f"Setting alpha to {selected_alpha=} -> {MULT}*Max cossine similarity: ")
                args.alpha = selected_alpha



            merged_tv = get_merged_tv(copy.deepcopy(tvs_to_merge), 
                                        args)
            
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
            
            r = eval_given_dataset(image_encoder, target_dataset, args.dataset, args)# * 100
            results_dict[f"merging/{domain_str}/{args.merge_fn}_{scaling_factor}"] = r

            # if args.debug:
            #     break 
        
        
        flat_list = list(results_dict.values())
        avg_top1_results = np.round(np.mean([x['top1'] for x in flat_list]).item(), 3).item() * 100
        avg_balacc_results = np.round(np.mean([x['bal_acc'] for x in flat_list]).item(), 3).item() * 100
        print(f"Avg. Top1: {avg_top1_results}%...")
        print(f"Avg. Bal.Acc: {avg_balacc_results}%...")
        pprint(results_dict)
        
        if args.debug:
            break
        else:
            wandb.log(results_dict)

    if not args.debug:
        wandb.finish()
