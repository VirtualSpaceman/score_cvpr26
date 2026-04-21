import numpy as np 
from src.merging.task_vectors import TaskVector, merge_rnd_mix, merge_max_abs, structured_randmix


def get_inter_range(merge_fn: str, use_zero_one_range: bool):
    if merge_fn == 'pcb':
        # aug_range = np.linspace(0.8, 2.5, endpoint=True, num=18) # nlp / mtl 
        aug_range = np.linspace(0.1, 2.5, endpoint=True, num=25)
    elif merge_fn in ['ties', 'dare_ties']: 
        # aug_range = np.linspace(0.8, 1.8, endpoint=True, num=11) # Ties original -> NLP / MTL 
        aug_range = np.linspace(0.1, 1.8, endpoint=True, num=18)
    elif merge_fn in ['tsv']:   
        # aug_range = np.linspace(0.6, 1.5, endpoint=True, num=10) # paper fig 4
        aug_range = np.linspace(0.5, 2.0, endpoint=True, num=16)
    elif merge_fn in ['isoc']:
        # aug_range = np.linspace(0.1, 1.5, endpoint=True, num=15) 
        aug_range = np.linspace(0.1, 2.0, endpoint=True, num=20)
    elif merge_fn in ['saliency']:
        aug_range = np.linspace(0.5, 2.0, endpoint=True, num=16)
    elif merge_fn in ['saliency_spectrum']:
        aug_range = np.linspace(0.5, 2.0, endpoint=True, num=16)
    elif merge_fn in ['saliency_svd', 'ours_v1']:
        aug_range = np.linspace(0.5, 1.5, endpoint=True, num=11)
    else:
        aug_range = np.linspace(0.1, 1.0, endpoint=True, num=10) 
    
    if use_zero_one_range:
        aug_range = np.linspace(0.1, 1.0, endpoint=True, num=10) 

    print(f"[{merge_fn=}] - Coefficient range: {aug_range}")
    return aug_range

def new_augment(merged_task_vector: TaskVector, args):
    new_task_vectors = []
    aug_range = get_inter_range(merge_fn=args.merge_fn, 
                                use_zero_one_range=args.use_zero_one_range)
    # Check if it is not False 
    if args.filter_coefs:
        print("Using filtered coefficients...")
        aug_range = args.new_coef_range

    print(f"[{args.merge_fn=}] [{args.pooling_fn}] - Coefficient range: {aug_range}")
    for coef in aug_range:
        new_task_vectors.append(merged_task_vector * coef)
    
    merged_task_vector = pooling_vectors(new_task_vectors, args.pooling_fn)

    return merged_task_vector

def plus_augment(merged_task_vector: TaskVector, task_vectors: list, args):
    print(f"tv before: {len(task_vectors)}")    
    aug_range = get_inter_range(merge_fn=args.merge_fn, 
                                use_zero_one_range=args.use_zero_one_range)
    
    # Check if it is not False 
    if args.filter_coefs:
        print("Using filtered coefficients...")
        aug_range = args.new_coef_range
        
    print(f"[{args.merge_fn=}] [{args.pooling_fn=}] - Coefficient range: {aug_range}")
    for coef in aug_range:
        task_vectors.append(merged_task_vector * coef)
    
    print(f"tv after: {len(task_vectors)}")    
    
    # pooling task vectors
    merged_task_vector = pooling_vectors(task_vectors, args.pooling_fn) 
    return merged_task_vector

def pooling_vectors(augmented_vectors, pooling_fn):
    # new merged task vector 
    merged_task_vector = None 
    if pooling_fn in ['randmix', None]:
        merged_task_vector = merge_rnd_mix(augmented_vectors)
    elif pooling_fn == 'magmax':
        merged_task_vector = merge_max_abs(augmented_vectors)
    elif pooling_fn == 'struct_mix':
        merged_task_vector = structured_randmix(augmented_vectors)
    elif pooling_fn == 'avg':
        N = len(augmented_vectors)
        merged_task_vector = sum(augmented_vectors) * float(1./N)
    
    assert merged_task_vector is not None, "merged_task_vector must not be None." 

    return merged_task_vector   
