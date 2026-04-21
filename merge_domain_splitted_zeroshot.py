import torch
import os
import json 
from src.eval import eval_given_dataset, get_dataset
from src.args import parse_arguments
from src.datasets.registry import registry
from src.constants import get_dict_dataset_paths
from pprint import pprint


if __name__ == '__main__':    
    # Config
    args = parse_arguments()
    PATHS = get_dict_dataset_paths()
    args.batch_size *= 2 
    method = 'ind-ft'
    models = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']
    models = [
            'ViT-B-32', 
            #   'ViT-B-16', 
            #   'ViT-L-14'
              ]
    datasets = [ 'DomainNet', 'ImageNetR', 
                'PACS', 'RetinaDomains', 
                'FedISIC', 'OfficeHome', 
                'NICOpp', 'TerraIncognita'
                ]
    

    for model in models:
        args.model = model
        pretrained_checkpoint = f'checkpoints/{args.model}/zeroshot.pt'


        for dataset in datasets:
            results_summary = dict()
            results_summary[model] = dict()
            args.dataset = dataset
            args.data_location = PATHS[args.dataset]
            results_summary[args.model][args.dataset] = dict()
            
            # Set results path 
            results_path = f'./results/zeroshot/DIL/{args.model}/{args.dataset}'
            os.makedirs(results_path, exist_ok=True)

            # Filename 
            save_file = 'zeroshot_performances.json'

            if os.path.exists(os.path.join(results_path, save_file)):
                print(f"Skipping inferece for {model=} and {dataset=} because file exists.")
                continue 

            # Get classification class
            dataset_class = registry[args.dataset]
            name=f"merging-{args.dataset}-DIL-{method}"
            
            all_domains = dataset_class.default_domain_order
            image_encoder = torch.load(pretrained_checkpoint)
            preprocess_fn = image_encoder.val_preprocess
            
            # Iterate over domains 
            for target_idx, target_domain in enumerate(all_domains): 
                domain_str = dataset_class.BASE_CLASS.DOMAINS[target_domain] 
                print(f"\nEVAL: {args.dataset} - on domain : {domain_str}")    
                args.subset_config = {
                    'domains': [dataset_class.BASE_CLASS.DOMAINS[target_domain]],
                    'classes': dataset_class.BASE_CLASS.CLASSES,
                }
                    
                target_dataset = get_dataset(
                                args.dataset,
                                preprocess_fn,
                                location=args.data_location,
                                batch_size=args.batch_size,
                                subset_config=args.subset_config,
                            )
                r = eval_given_dataset(image_encoder, target_dataset, args.dataset, args)

                results_summary[args.model][args.dataset][args.subset_config['domains'][0]] = r


            
            with open(os.path.join(results_path, save_file), 'w') as json_f:
                json.dump(results_summary, json_f, indent=4)

            pprint(results_summary) 
