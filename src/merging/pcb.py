import torch 


def clamp(x, min_ratio=0, max_ratio=0):
    if len(x.size())==1:
        d = x.size(0)
        sorted_x, _ = torch.sort(x)
        min=sorted_x[int(d * min_ratio)]
        max=sorted_x[int(d * (1-max_ratio)-1)]
    else:
        d = x.size(1)
        sorted_x, _ = torch.sort(x, dim=1)
        min=sorted_x[:, int(d * min_ratio)].unsqueeze(1)
        max=sorted_x[:, int(d * (1-max_ratio)-1)].unsqueeze(1)
    clamped_x= torch.clamp(x, min, max)
    return clamped_x

def act(x):
    y = torch.tanh(x)  # x**7; torch.relu(x)
    return y

def normalize(x, dim=0):
    min_values, _ = torch.min(x, dim=dim, keepdim=True)
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    y = (x - min_values) / (max_values - min_values)
    return y


def PCB(all_checks, pcb_ratio):
    n, d = all_checks.shape    
    all_checks_abs = clamp(torch.abs(all_checks), min_ratio=0.01, max_ratio=0.01)
    clamped_all_checks = torch.sign(all_checks)*all_checks_abs
    all_checks_normalized = torch.sign(all_checks) * normalize(all_checks_abs, dim=1)
    intra = normalize(all_checks_abs, 1)**2
    intra = torch.exp(n*intra)
    inter_score = []
    for i in range(n):
        score_i = torch.tanh(all_checks_normalized[i] * all_checks_normalized)
        inter_i = torch.sum(score_i, dim=0)
        inter_score.append(inter_i)
        
    inter = torch.vstack(inter_score)
    balancing = intra * inter
    scale = normalize(clamp(balancing, 1-pcb_ratio, 0), dim=1)
    return clamped_all_checks, scale

def get_pcb_inference_weights(clamp_tvs, scale, scaling_factor):
    num_vectors, _ = clamp_tvs.shape
    lams = num_vectors * [scaling_factor]
    tvs = clamp_tvs * torch.tensor(lams).unsqueeze(1)
    pcb_tv = torch.sum(tvs * scale, dim=0) / torch.clamp(torch.sum(scale, dim=0), min=1e-12)
    return pcb_tv