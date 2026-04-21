import numpy as np
import torch
import json 
import os
import copy 
from src.merging.task_vectors import TaskVector
from src.args import parse_arguments
from src.datasets.registry import registry
from src.eval import eval_get_features_or_logits, get_dataset
from sklearn.metrics import balanced_accuracy_score
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
    args.merge_fn = 'avg_ensemble'

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

        # Ensemble checkpooints
        tvs_to_ensemble = [tv for (idx, tv) in enumerate(task_vectors) if idx != target_idx]

        x_logits, y_true = None, None

        for tv in tvs_to_ensemble:
            # Get the fine-tuned checkpoint
            img_encoder = tv.apply_to(pretrained_checkpoint, scaling_coef=1.0)

            _logits, _y = eval_get_features_or_logits(img_encoder,
                                                subdomain_dataset, 
                                                args.dataset,
                                                args,
                                                logits=True)
            
            if args.debug: 
                print(f"[DEBUG] {_logits.shape=} / {_y.shape=}")
            # Here, we consider average logits ensemble 
            if x_logits is None:
                x_logits = _logits
            else: 
                x_logits += _logits
            
            if y_true is None:
                y_true = _y

        # Divide by the number of models within ensamble to get the average logits
        x_logits /= len(tvs_to_ensemble)  

        # Calculate acc and balanced acc 
        dict_metrics = dict()

        # calculate standard acc 
        y_preds = x_logits.argmax(dim=1, keepdim=True).to(args.device) 
        dict_metrics['top1'] = y_preds.eq(y_true.view_as(y_preds)).sum().item() / y_true.size(0)
        dict_metrics['bal_acc'] =  balanced_accuracy_score(y_true.cpu(), y_preds.cpu()).item()
        print(f"Ensemble performance for subdomain: {domain_str} -> {dict_metrics=}")
        
        results_dict[first_key][domain_str][f"acc_{scaling_factor}"] = dict_metrics
        
        if args.debug: 
            break 
            
    
    name=f"merging-{args.model}-{args.dataset}-DIL-{method}"
    json_filename = f"{name}.json"
    json_path = os.path.join(save_path, json_filename)

    if not args.debug: 
        with open(json_path, 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)