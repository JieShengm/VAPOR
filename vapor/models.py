import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from .utils import init_vae_weights

class TransportOperator(nn.Module):
    def __init__(self, latent_dim: int, n_dynamics: int, tau: float = 0.75,
                 gate_mode: str = "sigmoid_norm"):  # 'softmax' | 'sigmoid_norm' | 'sigmoid'
        super().__init__()
        self.latent_dim = latent_dim
        self.n_dynamics = n_dynamics
        self.gate_mode  = gate_mode
        
        self.Psi = nn.ParameterList([
            nn.Parameter(torch.empty(latent_dim, latent_dim))
            for _ in range(n_dynamics)
        ])
        for psi in self.Psi:
            # nn.init.orthogonal_(psi, gain=1e-3)
            nn.init.orthogonal_(psi, gain=1.0)
        self.gate_tokens = nn.Parameter(torch.randn(n_dynamics, latent_dim))
        nn.init.xavier_uniform_(self.gate_tokens)

        self.register_buffer("tau", torch.tensor(float(tau)))
        self.eps = 1e-8

        self.speed_head = nn.Sequential(
            nn.Linear(latent_dim, max(32, latent_dim // 4)),
            nn.LeakyReLU(),
            nn.Linear(max(32, latent_dim // 4), 1),
            nn.Softplus()
        )

    def unit_directions(self, z: torch.Tensor) -> torch.Tensor:
        U = torch.stack([z @ psi for psi in self.Psi], dim=1)         # (B, M, d)
        return U / (U.norm(dim=-1, keepdim=True).clamp_min(1e-6))     # (B, M, d)

    def gate_logits(self, Uhat: torch.Tensor) -> torch.Tensor:
        tok = F.normalize(self.gate_tokens, dim=-1, eps=1e-6)         # (M, d)
        logits = torch.einsum('bmd,md->bm', Uhat, tok)                # (B, M)
        return logits / self.tau.clamp_min(1e-6)

    def get_mixture_weights(self, Uhat: torch.Tensor) -> torch.Tensor:
        logits = self.gate_logits(Uhat)                               # (B, M)
        mode = self.gate_mode.lower()
        if mode == "softmax":
            pi = torch.softmax(logits, dim=1)
        elif mode == "sigmoid_norm":
            a  = torch.sigmoid(logits)
            pi = a / (a.sum(dim=1, keepdim=True) + self.eps)         
        elif mode == "sigmoid":
            pi = torch.sigmoid(logits) 
        else:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")
        return pi  

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        Uhat = self.unit_directions(z)                                # (B, M, d)
        pi   = self.get_mixture_weights(Uhat)                         # (B, M)
        v_dir = torch.einsum('bm,bmd->bd', pi, Uhat)                  # (B, d) 
        speed = self.speed_head(z).squeeze(-1)                        # (B,)
        return v_dir * speed.unsqueeze(-1)                            # (B, d)

    @torch.no_grad()
    def unit_velocity(self, z: torch.Tensor) -> torch.Tensor:
        Uhat = self.unit_directions(z)
        pi   = self.get_mixture_weights(Uhat)
        v    = torch.einsum('bm,bmd->bd', pi, Uhat)
        return F.normalize(v, dim=-1, eps=1e-6)

    def compute_velocities(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(torch.tensor(0.0, device=z.device), z)

    def sort_and_prune_psi(self, prune_threshold: float = None, relative: bool = False) -> None:
            # Compute norms
            norms = [psi.data.norm().item() for psi in self.Psi]
            max_norm = max(norms)
            
            # Sort by norms descending
            sorted_idxs = sorted(range(len(norms)), 
                                key=lambda i: norms[i], reverse=False)
            
            # Determine which channels to keep
            if prune_threshold is None:
                # Just sort, keep all
                keep = sorted_idxs
            else:
                if relative:
                    # Relative pruning: keep channels >= fraction * max_norm
                    keep = [i for i in sorted_idxs 
                           if norms[i] >= prune_threshold * max_norm]
                else:
                    # Absolute pruning: keep channels >= RMS threshold
                    D = self.latent_dim
                    n_elem = D * D
                    rms_norms = [norms[i] / math.sqrt(n_elem) for i in sorted_idxs]
                    keep = [sorted_idxs[j] for j, rms in enumerate(rms_norms) 
                           if rms >= prune_threshold]
            
            # Always keep at least one channel (the largest)
            if not keep:
                keep = [sorted_idxs[0]]
            
            # Rebuild
            self.Psi = nn.ParameterList([self.Psi[i] for i in keep])
            new_tokens = self.gate_tokens[keep].clone().detach()
            self.gate_tokens = nn.Parameter(new_tokens)
            self.n_dynamics = len(keep)
            
            # Print results
            if prune_threshold is None:
                print(f"Sorted {self.n_dynamics} channels by norm")
            else:
                if relative:
                    print(f"Pruned to {self.n_dynamics} channels (relative threshold: {prune_threshold})")
                else:
                    D = self.latent_dim
                    n_elem = D * D
                    kept_rms = [norms[i] / math.sqrt(n_elem) for i in keep]
                    print(f"Pruned to {self.n_dynamics} channels (RMS threshold: {prune_threshold})")
                    print(f"Kept RMS norms: {kept_rms}")

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, encoder_dims=None, decoder_dims=None):
        super().__init__()
        encoder_dims = encoder_dims or [2048, 1024, 512, 256, 128]
        decoder_dims = decoder_dims or list(reversed(encoder_dims))
        
        layers, prev = [], input_dim
        for h in encoder_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU()]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        self.encoder.apply(init_vae_weights)
        
        layers, prev = [], latent_dim
        for h in decoder_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU()]
            prev = h
        layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*layers)
        self.decoder.apply(init_vae_weights)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, z, mu, logvar

class VAPOR(nn.Module):
    def __init__(self, vae: VAE, transport_op: TransportOperator):
        super().__init__()
        self.vae = vae
        self.transport_op = transport_op

    def encode(self, x: torch.Tensor):
        return self.vae(x)
    
    def integrate(self, z0: torch.Tensor, t_span: torch.Tensor):
        z0_det = z0.detach()
        # reset NFE counter (optional)
        if hasattr(self.transport_op, "nfe"):
            print("Resetting NFE counter to zero.")
            self.transport_op.nfe.zero_()
        return odeint(self.transport_op, z0_det, t_span, 
                      method='rk4') 
                    #   method='dopri5', rtol=1e-4, atol=1e-5)

    def compute_velocities(self, z: torch.Tensor,) -> torch.Tensor:
        return self.transport_op.compute_velocities(z)

    # @staticmethod
    # def _zscore_coords(coords: torch.Tensor,
    #                 mean: torch.Tensor = None,
    #                 std: torch.Tensor = None,
    #                 eps: float = 1e-6) -> torch.Tensor:
    #     if mean is None:
    #         mean = coords.mean(dim=0, keepdim=True)
    #     if std is None:
    #         std = coords.std(dim=0, keepdim=True, unbiased=False)
    #     return (coords - mean) / std.clamp_min(eps)

    # def build_spatial_then_latent_graph(
    #     z: torch.Tensor,                       # (B, D)
    #     *,
    #     eps_z: float = None,                   # required if NO spatial
    #     coords: torch.Tensor = None,           # (B, 2/3) optional
    #     eps_xy: float = None,                  # required if spatial
    #     zscore_mode: str = "global",           # "global" | "batch"
    #     coords_mean: torch.Tensor = None,
    #     coords_std: torch.Tensor = None,
    #     min_samples: int = 5,
    #     k: int = 20,                           # top-k within candidates
    #     eps: float = 1e-6,
    # ):
    #     """
    #     Two-mode graph builder:

    #     (A) No spatial: latent radius graph using eps_z (old behavior).
    #     (B) Spatial: candidates from spatial radius (eps_xy), then top-k by latent distance.

    #     Returns:
    #     nbr_idx:  (B, Kmax) long
    #     nbr_mask: (B, Kmax) bool
    #     d2_latent:(B, B) float  (useful for gauss scores)
    #     """
    #     B = z.size(0)
    #     device = z.device

    #     # latent distances always computed (used for ordering and gauss)
    #     d_latent = torch.cdist(z, z)           # (B,B)
    #     d2_latent = d_latent.pow(2)

    #     use_spatial = (coords is not None) and (eps_xy is not None)

    #     if not use_spatial:
    #         # ---- Mode A: latent-only (old behavior) ----
    #         if eps_z is None:
    #             raise ValueError("eps_z must be provided when coords/eps_xy are not provided (latent-only mode).")

    #         mask = (d_latent <= float(eps_z))
    #         mask.fill_diagonal_(False)

    #         # min_samples filter
    #         deg = mask.sum(dim=1)
    #         noise = deg < min_samples
    #         mask[noise, :] = False

    #         # neighbor lists + optional top-k by latent distance
    #         neighbor_lists = []
    #         max_k = 0
    #         for i in range(B):
    #             idxs = mask[i].nonzero(as_tuple=True)[0]
    #             if k is not None and idxs.numel() > k:
    #                 order = torch.argsort(d_latent[i, idxs])  # latent ordering
    #                 idxs = idxs[order[:k]]
    #             neighbor_lists.append(idxs)
    #             max_k = max(max_k, idxs.numel())

    #         nbr_idx = torch.zeros((B, max_k), dtype=torch.long, device=device)
    #         nbr_mask = torch.zeros((B, max_k), dtype=torch.bool, device=device)
    #         for i, idxs in enumerate(neighbor_lists):
    #             n = idxs.numel()
    #             if n > 0:
    #                 nbr_idx[i, :n] = idxs
    #                 nbr_mask[i, :n] = True

    #         return nbr_idx, nbr_mask, d2_latent

    #     # ---- Mode B: spatial-first candidates, then top-k by latent distance ----
    #     if zscore_mode == "global":
    #         if coords_mean is None or coords_std is None:
    #             raise ValueError("coords_mean/std required for zscore_mode='global' when using spatial graph.")
    #         coords_z = (coords - coords_mean.to(device)) / coords_std.to(device).clamp_min(eps)
    #     elif zscore_mode == "batch":
    #         coords_z = _zscore_coords(coords, mean=None, std=None, eps=eps)
    #     else:
    #         raise ValueError(f"Unknown zscore_mode: {zscore_mode}")

    #     d_xy = torch.cdist(coords_z, coords_z)
    #     cand = (d_xy <= float(eps_xy))
    #     cand.fill_diagonal_(False)

    #     # min_samples on spatial candidates (drop isolated nodes)
    #     deg = cand.sum(dim=1)
    #     noise = deg < min_samples
    #     cand[noise, :] = False

    #     # within candidates pick top-k by latent distance
    #     neighbor_lists = []
    #     max_k = 0
    #     for i in range(B):
    #         idxs = cand[i].nonzero(as_tuple=True)[0]
    #         if idxs.numel() == 0:
    #             neighbor_lists.append(idxs)
    #             continue
    #         if k is not None and idxs.numel() > k:
    #             order = torch.argsort(d_latent[i, idxs])      # latent ordering INSIDE spatial candidates
    #             idxs = idxs[order[:k]]
    #         neighbor_lists.append(idxs)
    #         max_k = max(max_k, idxs.numel())

    #     nbr_idx = torch.zeros((B, max_k), dtype=torch.long, device=device)
    #     nbr_mask = torch.zeros((B, max_k), dtype=torch.bool, device=device)
    #     for i, idxs in enumerate(neighbor_lists):
    #         n = idxs.numel()
    #         if n > 0:
    #             nbr_idx[i, :n] = idxs
    #             nbr_mask[i, :n] = True

    #     return nbr_idx, nbr_mask, d2_latent

    # @torch.no_grad()
    # def build_directed_paths_cos_gauss(
    #     self,
    #     z: torch.Tensor,                  # (B, D)
    #     v: torch.Tensor,                  # (B, D)
    #     nbr_idx: torch.Tensor,            # (B, K)
    #     nbr_mask: torch.Tensor,           # (B, K)
    #     d2_latent: torch.Tensor,          # (B, B)
    #     T: int,
    #     cos_threshold: float = 0.0,
    # ) -> torch.Tensor:
    #     """
    #     Build directed paths by choosing next neighbor maximizing:
    #       score = cos_stretched(dz, v_dir) * gauss_norm(||dz||)

    #     where dz = z_neighbor - z_current, v_dir = normalized v_current.
    #     """
    #     B, D = z.shape
    #     device = z.device
    #     paths = torch.zeros((B, T), dtype=torch.long, device=device)

    #     curr = torch.arange(B, device=device)
    #     paths[:, 0] = curr

    #     v_dir_all = F.normalize(v, dim=1, eps=1e-6)  # (B, D)

    #     for t in range(1, T):
    #         nbrs = nbr_idx[curr]             # (B, K)
    #         valid = nbr_mask[curr]           # (B, K)

    #         z_c = z[curr].unsqueeze(1)       # (B, 1, D)
    #         z_n = z[nbrs]                    # (B, K, D)
    #         diffs = z_n - z_c                # (B, K, D)

    #         vd = v_dir_all[curr].unsqueeze(1)  # (B, 1, D)
    #         cos = F.cosine_similarity(diffs, vd, dim=-1)  # (B, K) in [-1,1]
    #         cos = (cos + 1.0) / 2.0                        # [0,1]
    #         cos = cos.masked_fill(cos < float(cos_threshold), 0.0)

    #         # stretch per-row to [0,1]
    #         c_min = cos.min(dim=1, keepdim=True).values
    #         c_max = cos.max(dim=1, keepdim=True).values
    #         cos_stretched = (cos - c_min) / (c_max - c_min + 1e-18)

    #         # gaussian on latent distance
    #         d2 = d2_latent[curr.unsqueeze(1), nbrs]        # (B, K)
    #         d = torch.sqrt(d2.clamp_min(0.0))

    #         # sigma per row = median neighbor distance
    #         d_masked = d.masked_fill(~valid, float("nan"))
    #         sigma = torch.nanmedian(d_masked, dim=1, keepdim=True).values
    #         sigma = sigma.clamp_min(1e-6)

    #         gauss = torch.exp(-d2 / (2.0 * sigma * sigma))  # (B, K)
    #         g_max = gauss.max(dim=1, keepdim=True).values
    #         gauss_norm = gauss / (g_max + 1e-18)

    #         score = cos_stretched * gauss_norm
    #         score = score.masked_fill(~valid, float("-inf"))

    #         best = score.argmax(dim=1)  # (B,)
    #         nxt = nbrs[torch.arange(B, device=device), best]
    #         paths[:, t] = nxt
    #         curr = nxt

    #     return paths

    # def directed_graph_tcl_loss(
    #     self,
    #     z0: torch.Tensor,
    #     z_traj: torch.Tensor,
    #     eps_z: float,
    #     min_samples: int = 5,
    #     threshold: float = 0.0,   # kept for signature compatibility; used as cos_threshold here
    #     k: int = 20,
    #     coords: torch.Tensor = None,
    #     eps_xy: float = None,
    #     coords_mean: torch.Tensor = None,
    #     coords_std: torch.Tensor = None,
    #     zscore_mode: str = "global",
    # ):
    #     """
    #     TCL loss with optional spatial constraint in the neighbor graph.

    #     Returns:
    #       loss, paths, adj_idx, adj_mask
    #     """
    #     # Graph (neighbor list)
    #     adj_idx, adj_mask, d2_latent = self.build_adaptive_spatial_then_latent_graph(
    #         z=z0,
    #         eps_z=eps_z,
    #         min_samples=min_samples,
    #         k=k,
    #         coords=coords,
    #         eps_xy=eps_xy,
    #         coords_mean=coords_mean,
    #         coords_std=coords_std,
    #         zscore_mode=zscore_mode,
    #     )

    #     # Paths (directed)
    #     v0 = self.compute_velocities(z0)
    #     T = z_traj.size(0)

    #     # reuse 'threshold' as cos threshold (keeps old API)
    #     paths = self.build_directed_paths_cos_gauss(
    #         z=z0,
    #         v=v0,
    #         nbr_idx=adj_idx,
    #         nbr_mask=adj_mask,
    #         d2_latent=d2_latent,
    #         T=T,
    #         cos_threshold=threshold,
    #     )

    #     # Loss: match integrated z_traj[t] to the “walked” nodes on z0 manifold
    #     loss = torch.stack(
    #         [F.mse_loss(z_traj[t], z0[paths[:, t]]) for t in range(1, T)]
    #     ).mean()

    #     return loss, paths, adj_idx, adj_mask

    def build_radius_graph(self, z: torch.Tensor, eps: float, min_samples: int = 2, k: int = None):
        B = z.size(0)
        dists = torch.cdist(z, z)
        mask = (dists <= eps)
        mask.fill_diagonal_(False)
        deg = mask.sum(dim=1)
        noise = deg < min_samples
        mask[noise, :] = False
        neighbor_lists = []
        max_k = 0
        for i in range(B):
            idxs = mask[i].nonzero(as_tuple=True)[0]
            if k is not None and idxs.numel() > k:
                order = torch.argsort(dists[i, idxs])
                idxs = idxs[order[:k]]
            neighbor_lists.append(idxs)
            max_k = max(max_k, idxs.numel())
        nbr_idx = torch.zeros((B, max_k), dtype=torch.long, device=z.device)
        nbr_mask = torch.zeros((B, max_k), dtype=torch.bool, device=z.device)
        for i, idxs in enumerate(neighbor_lists):
            n = idxs.numel()
            if n > 0:
                nbr_idx[i, :n] = idxs
                nbr_mask[i, :n] = True
        return nbr_idx, nbr_mask

    @torch.no_grad()
    def build_directed_paths_avgv(self, z, v, nbr_idx, nbr_mask, T, cos_threshold=0.0):
        B, D = z.size()
        K = nbr_idx.size(1)
        paths = torch.zeros((B, T), dtype=torch.long, device=z.device)
        curr = torch.arange(B, device=z.device)
        paths[:, 0] = curr

        for t in range(1, T):
            nbrs = nbr_idx[curr]
            valid = nbr_mask[curr]
            z_n = z[nbrs]
            z_c = z[curr].unsqueeze(1)
            diffs = z_n - z_c
            v_nbrs = v[nbrs] * valid.unsqueeze(-1)
            sum_v = v_nbrs.sum(dim=1)
            counts = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
            v_avg = sum_v / counts
            v_dir = F.normalize(v_avg, dim=1, eps=1e-6).unsqueeze(1)
            cosines = F.cosine_similarity(diffs, v_dir, dim=-1)
            cos_norm = (cosines + 1) / 2
            cos_norm = cos_norm.masked_fill(cos_norm < cos_threshold, 0.0)
            c_min, _ = cos_norm.min(dim=1, keepdim=True)
            c_max, _ = cos_norm.max(dim=1, keepdim=True)
            cos_stretched = (cos_norm - c_min) / (c_max - c_min + 1e-18)
            d2 = torch.sum(diffs*diffs, dim=-1)
            d_nb = torch.sqrt(d2)
            sigma = d_nb.median(dim=1, keepdim=True).values
            gauss_K = torch.exp(-d2 / (2 * sigma*sigma))
            g_max, _ = gauss_K.max(dim=1, keepdim=True)
            gauss_norm = gauss_K / (g_max + 1e-18)
            score = cos_stretched * gauss_norm
            score = score.masked_fill(~valid, float('-inf'))
            best = score.argmax(dim=1)
            nxt = nbrs[torch.arange(B, device=z.device), best]
            paths[:, t] = nxt
            curr = nxt
        return paths

    def directed_graph_tcl_loss(self, z0: torch.Tensor, z_traj: torch.Tensor,
                                eps: float, min_samples: int = 5,
                                threshold: float = 0.0, k: int = 20) -> torch.Tensor:
        adj_idx, adj_mask = self.build_radius_graph(z0, eps, min_samples, k)
        v0 = self.compute_velocities(z0)
        T = z_traj.size(0)
        paths = self.build_directed_paths_avgv(z0, v0, adj_idx, adj_mask, T, threshold)
        loss = torch.stack([F.mse_loss(z_traj[t], z0[paths[:,t]]) for t in range(1, T)]).mean()
        return loss, paths, adj_idx, adj_mask

    def flag_direction_loss_graph(self, z0: torch.Tensor, v0: torch.Tensor,
                                  is_start: torch.BoolTensor, is_term: torch.BoolTensor,
                                  nbr_idx: torch.LongTensor, nbr_mask: torch.BoolTensor) -> torch.Tensor:
        z_nbrs = z0[nbr_idx]
        mask_f = nbr_mask.unsqueeze(-1)
        mean_n = (z_nbrs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        projs = (v0 * (mean_n - z0)).sum(dim=1)
        losses = []
        if is_start.any():
            losses.append(F.relu(-projs[is_start]).mean())
        if is_term.any():
            losses.append(F.relu(projs[is_term]).mean())
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=z0.device)