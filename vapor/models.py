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
            nn.init.orthogonal_(psi, gain=1e-3)
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
            norms = [psi.data.norm().item() for psi in self.Psi]
            max_norm = max(norms)

            sorted_idxs = sorted(range(len(norms)), 
                                key=lambda i: norms[i], reverse=True)
            
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
            
            if not keep:
                keep = [sorted_idxs[0]]
            
            self.Psi = nn.ParameterList([self.Psi[i] for i in keep])
            new_tokens = self.gate_tokens[keep].clone().detach()
            self.gate_tokens = nn.Parameter(new_tokens)
            self.n_dynamics = len(keep)
            
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
                      # method='dopri5', rtol=1e-4, atol=1e-5)

    def compute_velocities(self, z: torch.Tensor,) -> torch.Tensor:
        return self.transport_op.compute_velocities(z)

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