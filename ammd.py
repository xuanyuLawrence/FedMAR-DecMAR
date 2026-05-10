import os
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector as p2v
import math
import numpy as np

@torch.no_grad()
def param_drift_abs(local_model: nn.Module, global_model: nn.Module) -> torch.Tensor:
    """
    Absolute L2 drift between local and global parameters:
        H_t = ||theta_local - theta_global||_2
    Returns:
        torch scalar tensor on the current device.
    """
    device = next(local_model.parameters()).device
    tl = p2v([p.detach() for p in local_model.parameters()]).to(device, non_blocking=True)
    tg = p2v([p.detach() for p in global_model.parameters()]).to(device, non_blocking=True)
    return torch.linalg.vector_norm(tl - tg, ord=2)

@torch.no_grad()
def update_client_gamma(cid, client_gamma, local_model_final, global_model_sent_this_round, G_MAX, G_MIN):
    state = client_gamma[cid]
    # 1) compute same-reference drift H_t
    H_t = param_drift_abs(local_model_final, global_model_sent_this_round)  # float

    # 2) first time? just record, keep gamma=G_MAX (your rule)
    if state['H_prev'] is None:
        state['H_prev'] = H_t
        # gamma keeps as is (initially G_MAX)
        return state['gamma']

    # 3) ratio update with deadband and small step
    eps = 1e-12
    DEADBAND = 0.0
    ALPHA = 1.0
    r = H_t / (state['H_prev'] + eps)
    if abs(r - 1.0) < DEADBAND:
        r = 1.0
    gamma_prev = float(state['gamma'])
    gamma_target = gamma_prev * r
    gamma_next = (1.0 - ALPHA) * gamma_prev + ALPHA * gamma_target
    gamma_next = max(G_MIN, min(G_MAX, gamma_next))

    # 4) persist
    state['gamma'] = gamma_next
    state['H_prev'] = H_t
    return gamma_next

def cosine_value(progress: float, g_min: float, g_max: float) -> float:
    """
    Cosine annealing on progress p in [0, 1]:
        gamma = g_min + 0.5*(g_max - g_min)*(1 + cos(pi * p))
    """
    p = max(0.0, min(1.0, progress))
    return g_min + 0.5 * (g_max - g_min) * (1.0 + math.cos(math.pi * p))

def progress_from_count(count: int, c_cap: int) -> float:
    """
    Map selection count c -> progress p in [0, 1].
    Once c >= c_cap, progress saturates at 1.
    """
    if c_cap <= 0:
        return 1.0
    return min(1.0, max(0.0, count / float(c_cap)))

def update_gamma_by_cosine(client_gamma, sampled_clients,
                       g_max: float, g_min: float,  c_cap: int) -> None:
    """
    Call ONCE after aggregation each round.
    1) Increment 'count' ONLY for clients that participated in this round.
    2) Recompute gamma for ALL clients using cosine(progress(count/c_cap)).
    """
    # 1) bump counts for selected clients
    for cid in sampled_clients:
        st = client_gamma[cid]
        st['count'] = st['count'] + 1

    # 2) recompute gamma
    for cid, st in client_gamma.items():
        c = st['count']
        p = progress_from_count(c, c_cap=c_cap)               # in [0,1]
        st['gamma'] = float(cosine_value(p, g_min=g_min, g_max=g_max))

def median_pairwise_distance(x):
    with torch.no_grad():
        dist = torch.cdist(x, x, p=2)
        median = dist.flatten().median()
    return median.item()

def mean_pairwise_distance(x):
    with torch.no_grad():
        dist = torch.cdist(x, x, p=2)
        mean = dist.flatten().mean()
    return mean.item()

def gaussian_kernel(x, y, sigma=1.0):
    x = x.unsqueeze(1)  # [B, 1, D]
    y = y.unsqueeze(0)  # [1, B, D]
    dist = ((x - y) ** 2).sum(dim=2)
    return torch.exp(-dist / (2 * sigma ** 2))

def mmd_loss(x, y, sigma=1.0):
    # x, y: [B, D] — local and global embeddings
    K_xx = gaussian_kernel(x, x, sigma)
    K_yy = gaussian_kernel(y, y, sigma)
    K_xy = gaussian_kernel(x, y, sigma)

    # Remove diagonal for unbiased estimate
    B = x.size(0)
    B_y = y.size(0)
    loss = K_xx.sum() / (B * (B - 1)) + K_yy.sum() / (B_y * (B_y - 1)) - 2 * K_xy.mean()
    return loss

def mix_weights(randomClientIDs, data_len_ratios, distance_dict=None):

    a = np.array([float(data_len_ratios[c]) for c in randomClientIDs], dtype=np.float64)

    if not distance_dict:
        return data_len_ratios

    d = np.array([float(distance_dict[c]) for c in randomClientIDs], dtype=np.float64)

    med = float(np.median(d))
    s = 1 / (1 + d / med)
    c = s / (s.mean())

    w_raw = a * c
    w = w_raw / w_raw.sum()
    # print(d)
    # print(a)
    # print(s)
    # print(c)
    # print(w_raw)
    # print(w)
    return {cid: float(w[i]) for i, cid in enumerate(randomClientIDs)}

@torch.no_grad()
def pool_latent(latent: torch.Tensor) -> torch.Tensor:
    """
    latent: [B, L, D] / [N, D] / [D]
    return: [D] (float32, CPU)
    """
    x = latent
    if x.dim() == 3:   # [B, L, D]
        x = x.mean(dim=(0, 1))
    elif x.dim() == 2: # [N, D]
        x = x.mean(dim=0)
    elif x.dim() == 1: # [D]
        pass
    else:
        raise ValueError(f"Unexpected latent shape: {latent.shape}")
    return x.detach().float().cpu()

@torch.no_grad()
def cosine_distance_vec(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> float:
    """
    u, v: [D] CPU/float32
    return: float，1 - cos(u, v)
    """
    un = u / (u.norm(p=2) + eps)
    vn = v / (v.norm(p=2) + eps)
    sim = float(torch.dot(un, vn).item())
    return 1.0 - sim



@torch.no_grad()
def geometric_median_tensor(latents, iters: int = 10, eps: float = 1e-9):
    """
    Compute geometric median across a list of tensors with identical shape [N, L, D].
    Weiszfeld algorithm, vectorized over positions (N,L).

    Args:
        latents: List[Tensor], each of shape [N, L, D], same dtype/device ideally.
        iters:   Number of Weiszfeld iterations (10 is usually enough).
        eps:     Small constant to avoid division by zero.

    Returns:
        Tensor of shape [N, L, D] (float32, same device as inputs[0]).
    """
    assert len(latents) > 0, "Empty latents list"
    # Stack -> [K, N, L, D]
    X = torch.stack([x.float() for x in latents], dim=0)  # use fp32 for stability
    device = X.device

    # Init with arithmetic mean (good starting point)
    m = X.mean(dim=0)  # [N, L, D]

    for _ in range(iters):
        # distances per sample to current median, vector L2 over D
        diff = X - m                      # [K, N, L, D]
        dist = diff.norm(dim=-1)          # [K, N, L]
        w = 1.0 / (dist.clamp_min(eps))   # [K, N, L]  weights = 1 / ||x - m||

        # weighted update
        w_exp = w.unsqueeze(-1)           # [K, N, L, 1]
        m = (w_exp * X).sum(dim=0) / (w_exp.sum(dim=0).clamp_min(eps))  # [N, L, D]

    return m  # [N, L, D]

@torch.no_grad()
def sphericalize(z, eps=1e-12):  # z: [..., D]
    return z / (z.norm(dim=-1, keepdim=True) + eps)

@torch.no_grad()
def mse_distance_vec(u: torch.Tensor, v: torch.Tensor) -> float:
    """
    u, v: [D] float32 CPU/CUDA
    """
    u = u.detach().float()
    v = v.detach().float()
    return float(torch.mean((u - v) ** 2).item())