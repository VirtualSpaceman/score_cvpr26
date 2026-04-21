import torch 
import numpy as np 

from src.merging.pcb import PCB
from src.merging.tsvm import tsv_merge
from src.merging.score import SCORE
from src.merging.isoc import iso_c, iso_cts
from src.merging.task_vectors import TaskVector, merge_max_abs, merge_rnd_mix
from src.merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict


def ties(task_vectors):
    # TIES merging
    reset_type = 'topk'
    reset_thresh = 20
    resolve = 'mass'
    merge = 'dis-mean'
    tv_flat_checks = torch.vstack([state_dict_to_vector(tv.vector) for tv in task_vectors])
    
    print(f"\nMerging with TIES merging: pruning {reset_type}-{reset_thresh}, resolve sign by {resolve}, merge by {merge}")
    
    merged_flat_tv = merge_methods(
        reset_type,
        tv_flat_checks,
        reset_thresh=reset_thresh,
        resolve_method=resolve,
        merge_func=merge,
    )
    merged_tv = vector_to_state_dict(
        merged_flat_tv, task_vectors[0].vector, remove_keys=[]
    )
    merged_tv = TaskVector(vector=merged_tv)
    
    return merged_tv

def pcb_merging(task_vectors):
    tv_flat_checks = torch.vstack([state_dict_to_vector(tv.vector) for tv in task_vectors])
    # https://github.com/duguodong7/pcb-merging/blob/5897888b5d24adfcd33e124793d113746e00f3ff/vision_source_code/pcb_ES.py#L103
    pcb_ratio = 0.05
    return PCB(tv_flat_checks, pcb_ratio=pcb_ratio)

import time

def get_merged_tv_timed(task_vectors, args):
    merged_tv = None 
    runtime = 0.0  # Variable to store the calculated runtime
    
    # Helper to calculate runtime for a single function call
    def exec_timed(func):
        start = time.time()
        result = func()
        end = time.time()
        return result, end - start

    # MagMax
    if args.merge_fn == 'magmax':
        merged_tv, runtime = exec_timed(lambda: merge_max_abs(task_vectors))
    # RandMix
    if args.merge_fn == 'randmix':
        merged_tv, runtime = exec_timed(lambda: merge_rnd_mix(task_vectors))
    # Task arithmetic
    if args.merge_fn in ['avg', 'dare_ta']:
        merged_tv, runtime = exec_timed(lambda: sum(task_vectors))
    # TIES Merging
    if args.merge_fn in ['ties', 'dare_ties']:
        merged_tv, runtime = exec_timed(lambda: ties(task_vectors))
    if args.merge_fn == 'breadcumbs':
        merged_tv, runtime = exec_timed(lambda: sum(task_vectors))
    # PCB
    if args.merge_fn == 'pcb':
        # returns a pair (tensor, scale)
        merged_tv, runtime = exec_timed(lambda: pcb_merging(task_vectors))
    # ISO-C
    if args.merge_fn == 'isoc':
        merged_tv, runtime = exec_timed(lambda: iso_c(task_vectors, args.device))
    # ISO-CTS
    if args.merge_fn == 'iso_cts':
        merged_tv, runtime = exec_timed(lambda: iso_cts(task_vectors, args))
    # TSV
    if args.merge_fn == 'tsv':
        merged_tv, runtime = exec_timed(lambda: tsv_merge(task_vectors, args.device))
    # SCORE
    if args.merge_fn.startswith('ours_v'):
        merged_tv, runtime = exec_timed(lambda: SCORE(task_vectors=task_vectors,
                                                           args=args))
    
    assert merged_tv is not None, "Merged Vector should not be None!"
    
    return merged_tv, runtime

def get_merged_tv(task_vectors, args):
    merged_tv = None 
    # MagMax
    if args.merge_fn == 'magmax':
        merged_tv = merge_max_abs(task_vectors)
    if args.merge_fn == 'randmix':
        merged_tv = merge_rnd_mix(task_vectors)
    # Task arithmetic
    if args.merge_fn in ['avg', 'dare_ta']:
        merged_tv = sum(task_vectors)
    if args.merge_fn in ['ties', 'dare_ties']:
        merged_tv = ties(task_vectors)
    if args.merge_fn == 'breadcumbs':
        merged_tv = sum(task_vectors)
    if args.merge_fn == 'pcb':
        # returns a pair (tensor, scale)
        merged_tv = pcb_merging(task_vectors)
    if args.merge_fn == 'isoc':
        merged_tv = iso_c(task_vectors, args.device)
    if args.merge_fn == 'iso_cts':
        merged_tv = iso_cts(task_vectors, args)
    if args.merge_fn == 'tsv':
        merged_tv = tsv_merge(task_vectors, args.device)
    if args.merge_fn.startswith('ours_v'):
        merged_tv = SCORE(task_vectors=task_vectors,
                               args=args)


    assert merged_tv is not None, "Merged Vector should not be None!"
    
    return merged_tv

