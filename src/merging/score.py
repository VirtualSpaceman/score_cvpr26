import math
import torch 
from .task_vectors import TaskVector


def trimm_avg_all_off(matrix, num_tasks, is_medical_task=False, 
                      diag=True, off_diag=True, trim=True):

    #  Zero out diagonal elements
    off_diag_elements = matrix.clone()
    off_diag_elements.diagonal(dim1=-1, dim2=-2).zero_()

    off_diag_mean = off_diag_elements.mean()
    off_diag_std = off_diag_elements.std()

    MULT = 1.96
    if is_medical_task:
        threshold = MULT * off_diag_std  
    else: 
        threshold = MULT * off_diag_std/num_tasks 
         
    diag_only = torch.diag(torch.diag(matrix))

    trimmed_mat = 0
    if diag:
        trimmed_mat += diag_only 
    
    if off_diag: 

        if trim:
            off_diag_mask = (off_diag_elements - off_diag_mean).abs() <= threshold
            trimm_off_diag_elements = off_diag_elements * off_diag_mask
            trimmed_mat += trimm_off_diag_elements
        else: 
            trimmed_mat += off_diag 

    return trimmed_mat


def SCORE(task_vectors, args):   

    device = args.device 
    sv_reduction = 1 / len(task_vectors)

    is_medical_task = args.dataset in ['FedISIC', 'RetinaDomains']
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, task_vector in enumerate(task_vectors):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        # print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                

                # if any([args.merge_fn.endswith(x) for x in ['v1', 'v2']]):
                #     u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                #     u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                #     U_new = u_u @ v_u
                #     V_new = u_v @ v_v
                # elif any([args.merge_fn.endswith(x) for x in ['v3', 'v4']]): 
                #     U_new = sum_u
                #     V_new = sum_v
                # else: 
                #     raise ValueError
                # new_sigma = 0
                
                u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                U_new = u_u @ v_u
                V_new = u_v @ v_v
                
                new_sigma = 0

                # Compute the new sigma 
                for i, task_vector in enumerate(task_vectors):
                    current_change_basis = U_new.T @ task_vector.vector[key].to(device) @ V_new.T

                    # include diag and off diag trim
                    if args.merge_fn.endswith('v1'):
                        new_sigma += trimm_avg_all_off(current_change_basis, 
                                                    num_tasks=len(task_vectors),
                                                    is_medical_task=is_medical_task,
                                                    diag=True,
                                                    off_diag=True,
                                                    trim=True)
                    # Include diag only 
                    if args.merge_fn.endswith('v2'):
                        print("Including only diagonal matrix...")
                        new_sigma += trimm_avg_all_off(current_change_basis, 
                                                    num_tasks=len(task_vectors),
                                                    is_medical_task=is_medical_task,
                                                    diag=True,
                                                    off_diag=False,
                                                    )
                    # Include off diag only 
                    if args.merge_fn.endswith('v3'):
                        print("Including off-diagonal elements only (no prune) ...")
                        new_sigma += trimm_avg_all_off(current_change_basis, 
                                                    num_tasks=len(task_vectors),
                                                    is_medical_task=is_medical_task,
                                                    diag=False,
                                                    off_diag=True,
                                                    )
                        
                    # include full matrix -> diag and off diag (no trim)
                    if args.merge_fn.endswith('v4'):
                        new_sigma += trimm_avg_all_off(current_change_basis, 
                                                    num_tasks=len(task_vectors),
                                                    is_medical_task=is_medical_task,
                                                    diag=True,
                                                    off_diag=True,
                                                    trim=False
                                                    )
                    

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        U_new,
                        new_sigma,
                        V_new,
                    )
                )


    return TaskVector(vector=new_vector)