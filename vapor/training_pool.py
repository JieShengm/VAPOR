import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 假设的基础工具函数占位
def resolve_device(config):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Utils ---
def set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = float(lr)

def get_optimizer_lr(opt: torch.optim.Optimizer) -> float:
    return float(opt.param_groups[0]["lr"])

class _WithIndex(Dataset):
    def __init__(self, base: Optional[Dataset]):
        self.base = base
    def __len__(self):
        return 0 if self.base is None else len(self.base)
    def __getitem__(self, idx: int):
        item = self.base[idx]
        if isinstance(item, (tuple, list)):
            return (*item, idx)
        return (item, idx)

def pick_pool_size_by_N(N: int, lo: int = 500, hi: int = 1000) -> int:
    if N <= 1500:
        base = lo
    elif N >= 5000:
        base = hi
    else:
        base = int(lo + (N - 1500) * ((hi - lo) / (5000 - 1500)))
    return max(50, min(base, N - 1))

# --- Core Components ---

@dataclass
class RollingPool:
    """环形滚动 pool：修复了 batch > pool_size 时的逻辑。"""
    pool_size: int
    dim: int
    device: torch.device
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        self.z = torch.zeros((self.pool_size, self.dim), device=self.device, dtype=self.dtype)
        self.v = torch.zeros((self.pool_size, self.dim), device=self.device, dtype=self.dtype)
        self.ptr = 0
        self.filled = 0

    @torch.no_grad()
    def add(self, z_batch: torch.Tensor, v_batch: torch.Tensor) -> None:
        z_batch, v_batch = z_batch.detach(), v_batch.detach()
        B = z_batch.size(0)
        if B == 0: return
        
        # 处理 B > pool_size 的情况
        if B > self.pool_size:
            z_batch = z_batch[-self.pool_size:]
            v_batch = v_batch[-self.pool_size:]
            B = self.pool_size

        end = self.ptr + B
        if end <= self.pool_size:
            self.z[self.ptr:end] = z_batch
            self.v[self.ptr:end] = v_batch
        else:
            first_part = self.pool_size - self.ptr
            self.z[self.ptr:] = z_batch[:first_part]
            self.v[self.ptr:] = v_batch[:first_part]
            self.z[:end % self.pool_size] = z_batch[first_part:]
            self.v[:end % self.pool_size] = v_batch[first_part:]
        
        self.ptr = (self.ptr + B) % self.pool_size
        self.filled = min(self.pool_size, self.filled + B)

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        M = int(self.filled)
        if M <= 0: raise RuntimeError("Pool is empty.")
        return self.z[:M], self.v[:M]

@torch.no_grad()
def _build_pool_knn_graph(pool_z: torch.Tensor, K: int) -> torch.Tensor:
    M = pool_z.size(0)
    if M <= 1: return torch.zeros((M, 0), dtype=torch.long, device=pool_z.device)
    K = max(1, min(K, M - 1))
    # 使用 chunk 计算以节省大 Pool 时的显存
    dists = torch.cdist(pool_z, pool_z)
    return dists.topk(K + 1, dim=1, largest=False).indices[:, 1:]

@torch.no_grad()
def _knn_batch_to_pool(z0: torch.Tensor, pool_z: torch.Tensor, K: int) -> torch.Tensor:
    M = pool_z.size(0)
    K = max(1, min(K, M - 1))
    dists = torch.cdist(z0, pool_z)
    return dists.topk(K, dim=1, largest=False).indices

@torch.no_grad()
def _sample_from_q_top_p(q: torch.Tensor, nbrs: torch.Tensor, top_p: float = 0.9):
    B, K = q.shape
    q_sorted, idx_sorted = torch.sort(q, dim=1, descending=True)
    cdf = torch.cumsum(q_sorted, dim=1)
    keep = cdf <= top_p
    keep[:, 0] = True 
    q_nuc = q_sorted * keep.float()
    q_nuc = q_nuc / q_nuc.sum(dim=1, keepdim=True).clamp_min(1e-12)
    
    sampled_pos = torch.multinomial(q_nuc, num_samples=1).squeeze(1)
    sampled_k = idx_sorted[torch.arange(B, device=q.device), sampled_pos]
    return nbrs[torch.arange(B, device=q.device), sampled_k]

@torch.no_grad()
def build_directed_soft_targets_rollout_poolspace(
    z0: torch.Tensor, pool_z: torch.Tensor, pool_v: torch.Tensor,
    nbr_idx_pool: torch.Tensor, T: int, K: int, v0: torch.Tensor,
    cos_threshold: float = 0.0, tau_q: float = 0.25, top_p: float = 0.8, min_dist: float = 1e-6,
):
    device = pool_z.device
    B, D = z0.shape
    mu_targets = torch.zeros((B, T, D), device=device)
    curr = None

    for t in range(1, T):
        if t == 1:
            nbrs = _knn_batch_to_pool(z0, pool_z, K=K)
            z_c = z0.unsqueeze(1)
            v_dir = F.normalize(v0, dim=1, eps=1e-6).unsqueeze(1)
        else:
            nbrs = nbr_idx_pool[curr]
            z_c = pool_z[curr].unsqueeze(1)
            v_dir = F.normalize(pool_v[curr], dim=1, eps=1e-6).unsqueeze(1)

        diffs = pool_z[nbrs] - z_c
        cosines = F.cosine_similarity(diffs, v_dir, dim=-1)
        cos_norm = (cosines + 1.0) / 2.0
        
        if cos_threshold > 0.0:
            cos_norm = cos_norm.masked_fill(cos_norm < cos_threshold, 0.0)

        c_min, c_max = cos_norm.min(1, keepdim=True).values, cos_norm.max(1, keepdim=True).values
        cos_stretched = (cos_norm - c_min) / (c_max - c_min + 1e-12)

        d2 = (diffs**2).sum(dim=-1)
        sigma = d2.sqrt().median(dim=1, keepdim=True).values.clamp_min(1e-8)
        g_max = gauss_norm.max(dim=1, keepdim=True).values
        gauss_norm = gauss_norm / (g_max + 1e-12)

        small = d2 < min_dist**2
        all_bad = small.all(dim=1)
        small[all_bad, 0] = False
        score = score.masked_fill(small, -1e9)
        q = torch.softmax(score / max(tau_q, 1e-6), dim=1)

        curr = _sample_from_q_top_p(q, nbrs, top_p=top_p)
        mu_targets[:, t] = pool_z[curr]

    return mu_targets

# --- Training Loop ---

def train_model(
    model, dataset, config=None, split_train_test=True, test_size=0.2,
    save_dir="vapor/out/", exp_name="run", verbose=True, **kwargs
):
    # 配置初始化
    device = resolve_device(config)
    model.to(device)
    
    # 确保 DataLoader 无限循环的工程实现
    def endless_loader(l):
        while True:
            for b in l: yield b

    train_dataset = _WithIndex(dataset) # 简化逻辑：这里直接用原数据集
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    loader_iter = endless_loader(train_loader)

    # 优化器分配
    vae_params = [p for n, p in model.named_parameters() if "vae." in n]
    other_params = [p for n, p in model.named_parameters() if "vae." not in n]
    opt_vae = torch.optim.AdamW(vae_params, lr=config.lr)
    opt_to = torch.optim.AdamW(other_params, lr=config.lr)

    # Pool 初始化
    pool = RollingPool(pool_size=pick_pool_size_by_N(len(dataset)), dim=model.vae.latent_dim, device=device)
    
    history = []
    model.train()

    for global_step in range(config.total_steps):
        batch = next(loader_iter)
        x = batch[0].to(device, non_blocking=True)
        
        # VAE Pass
        recon, z0, mu0, logvar0 = model.encode(x)
        recon_loss = F.mse_loss(recon, x)
        kl_loss = (-0.5 * (1 + logvar0 - mu0.pow(2) - logvar0.exp())).mean()
        vae_loss = recon_loss + config.beta * kl_loss

        # Transport Pass
        loss_transport = torch.tensor(0.0, device=device)
        enable_transport = (pool.filled >= 200) and (global_step > config.total_steps * 0.1)

        if enable_transport:
            pool_z, pool_v = pool.get()
            # 这里的 z_traj 必须与 other_params 挂钩 (即 model.integrate 内部要有参数)
            t_rand = torch.randint(1, config.t_max + 1, (1,)).item()
            z_traj = model.integrate(z0, torch.linspace(0, t_rand, t_rand + 1, device=device))
            
            # 生成目标（Target 是 detach 的，作为监督信号）
            with torch.no_grad():
                start_z = mu0 if config.graph_use_mu else z0
                start_v = model.compute_velocities(start_z)
                mu_targets = build_directed_soft_targets_rollout_poolspace(
                    start_z, pool_z, pool_v, _build_pool_knn_graph(pool_z, config.graph_k),
                    T=z_traj.size(0), K=config.graph_k, v0=start_v
                )
            
            traj_loss = F.mse_loss(z_traj[1:], mu_targets[:, 1:].transpose(0, 1))
            loss_transport = config.traj_weight * traj_loss

        # Step
        opt_vae.zero_grad(set_to_none=True)
        opt_to.zero_grad(set_to_none=True)
        
        (vae_loss + loss_transport).backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_vae.step()
        if enable_transport: opt_to.step()

        # 更新 Pool
        with torch.no_grad():
            pool.add(mu0 if config.graph_use_mu else z0, model.compute_velocities(mu0 if config.graph_use_mu else z0))

        if verbose and global_step % 100 == 0:
            print(f"Step {global_step} | Recon: {recon_loss.item():.4f} | Traj: {loss_transport.item():.4f}")

    return model