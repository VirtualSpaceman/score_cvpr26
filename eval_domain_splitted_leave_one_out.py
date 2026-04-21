import numpy as np
import torch
import json 
import os
import copy 
from src.merging.task_vectors import TaskVector
from src.args import parse_arguments
from src.datasets.registry import registry
from src.eval import get_dataset, eval_given_dataset
from src.constants import get_dict_dataset_paths, get_dict_epochs

# Config
args = parse_arguments()
pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'


if __name__ == '__main__':    
    PATHS = get_dict_dataset_paths()
    EPOCHS = get_dict_epochs()
    
    dataset_class = registry[args.dataset]
    method = 'seq-ft' if args.sequential_finetuning else 'ind-ft'
    args.epochs = EPOCHS[args.dataset]
    args.data_location = PATHS[args.dataset]

    # Always change the merge_fn for better compatibility 
    args.merge_fn = 'upperbound'

    # Get all domains and task vectors
    all_domains = dataset_class.default_domain_order
    task_vectors = []

    for task_idx, domain_idx in enumerate(all_domains):

        args.subset_config = {
            'domains': [dataset_class.BASE_CLASS.DOMAINS[domain_idx]],
            'classes': dataset_class.BASE_CLASS.CLASSES,
        }
        subset_config_id = dataset_class.BASE_CLASS.get_md5(args.subset_config)
        args.save = f'checkpoints/{args.model}/domain_incremental'
        ckpdir = os.path.join(args.save, args.dataset)
        ft_path = os.path.join(ckpdir, f'checkpoint_ep:{args.epochs}-lr:{args.lr}_{subset_config_id}.pt')
        
        tv = TaskVector(pretrained_checkpoint, ft_path)
        task_vectors.append(tv)


    # setup dict to log results 
    results_dict = dict()
    save_path = os.path.join("./results", "merging", args.model, args.dataset, args.merge_fn, method)
    os.makedirs(save_path, exist_ok=True)

    first_key = f"{args.merge_fn}_{args.dataset}"
    results_dict[first_key] = dict()
    # merge_fn -> dataset -> domains -> -> acc

    # Get scaling factors 
    scaling_factor = 1.0
    scaling_factor = np.round(scaling_factor, 2)

    base_model = torch.load(pretrained_checkpoint)
    base_model = task_vectors[0].apply_to(pretrained_checkpoint, scaling_coef=0.0)
    preprocess_fn = base_model.val_preprocess
    del base_model

    # Evaluate the expert model only!
    for target_idx, target_domain in enumerate(all_domains): 
        # leave the target domain out and marge the remaining 
        domain_str = dataset_class.BASE_CLASS.DOMAINS[target_domain]
        results_dict[first_key][domain_str] = dict()

        args.subset_config = {
            'domains': [domain_str],
            'classes': dataset_class.BASE_CLASS.CLASSES,
        }


        subdomain_dataset = get_dataset(
                            args.dataset,
                            preprocess_fn,
                            location=args.data_location,
                            batch_size=args.batch_size,
                            subset_config=args.subset_config,
                        )

        # Ensemble checkpoints
        tv_eval = task_vectors[target_idx]


        eval_model = tv_eval.apply_to(pretrained_checkpoint, scaling_coef=1.0)


        dict_metrics = eval_given_dataset(image_encoder=eval_model, 
                                     dataset=subdomain_dataset, 
                                     dataset_name=args.dataset,
                                     args=args)


        print(f"Expert performance for subdomain: {domain_str} -> {dict_metrics=}")
        
        results_dict[first_key][domain_str][f"acc_{scaling_factor}"] = dict_metrics
        
        if args.debug: 
            break 
            
    
    name=f"merging-{args.model}-{args.dataset}-DIL-{method}"
    json_filename = f"{name}.json"
    json_path = os.path.join(save_path, json_filename)

    if not args.debug: 
        with open(json_path, 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
    