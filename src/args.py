import os
import random
import numpy as np
import argparse

import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # DATASETS
    parser.add_argument(
        "--data-location",
        type=str,
        default="/hadatasets/levy/datasets/",
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )

    # MODEL/TRAINING
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    
    # LOAD/SAVE PATHS
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default='checkpoints/ViT-B-16/cachedir/open_clip',
        help='Directory for caching models from OpenCLIP'
    )
    
    # OTHER    
    parser.add_argument(
        '--seed',
        default=5,
        type=int
    )
    parser.add_argument(
        "--wandb_entity_name",
        type=str,
        default="levy-gu"
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
    )

    parser.add_argument('--merge-fn', 
                    default='magmax',
                    choices=['magmax', 'randmix', 'avg', 'ties', 'modelsoup', 
                             'model_stock', 'breadcumbs', 
                             'isoc', 
                             'iso_cts', 
                             'tsv',  
                             'ours_v1', 'ours_v2', 'ours_v3', 'ours_v4',
                             'pcb', 
                             ],
                    type=str)


    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(parsed_args.seed)

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]

    return parsed_args
