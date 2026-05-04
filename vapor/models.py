import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from .utils import init_vae_weights

# ============================================================
# Transport Operator
# ============================================================

class TransportOperator(nn.Module):
    def __init__(self, latent_dim: int, n_dynamics: int, gate_temperature: float = 0.75,
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

        self.register_buffer("gate_temperature", torch.tensor(float(gate_temperature)))
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
        return logits / self.gate_temperature.clamp_min(1e-6)

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
        speed = self.speed_head(z).squeeze(-1)
        # speed = torch.tanh(speed / 3.0) * 3.0# (B,)
        # speed = torch.clamp(speed, max=2.0)
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

# ============================================================
# VAE
# ============================================================

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, encoder_dims=None, decoder_dims=None):
        super().__init__()
        encoder_dims = encoder_dims or [2048, 1024, 512, 256, 128]
        decoder_dims = decoder_dims or list(reversed(encoder_dims))
        
        layers, prev = [], input_dim
        for h in encoder_dims:
            layers += [nn.Linear(prev, h),
                       nn.LayerNorm(h),  
                       nn.LeakyReLU()]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        
        layers, prev = [], latent_dim
        for h in decoder_dims:
            layers += [nn.Linear(prev, h),
                       nn.LayerNorm(h),  
                       nn.LeakyReLU()]
            prev = h
        layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*layers)
        self.decoder.apply(init_vae_weights)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # logvar = -3.75 + 2.25 * torch.tanh(logvar / 5.0)
        return mu, logvar

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

# ============================================================
# VAPOR
# ============================================================

class VAPOR(nn.Module):
    def __init__(self, vae: VAE, transport_op: TransportOperator):
        super().__init__()
        self.vae = vae
        self.transport_op = transport_op

    def encode(self, x: torch.Tensor):
        return self.vae(x)
    
    # def integrate(self, z0: torch.Tensor, t_span: torch.Tensor):
    #     z0_det = z0.detach()
    #     # reset NFE counter (optional)
    #     if hasattr(self.transport_op, "nfe"):
    #         print("Resetting NFE counter to zero.")
    #         self.transport_op.nfe.zero_()
    #     return odeint(self.transport_op, z0_det, t_span, 
    #                   method='rk4') 

    def integrate(self, z0: torch.Tensor, t_span: torch.Tensor,
              dt: float = 1.0):
        z0_det = z0.detach()
        print(f"Integrating with dt={dt}...")
        # reset NFE counter (optional)
        if hasattr(self.transport_op, "nfe"):
            print("Resetting NFE counter to zero.")
            self.transport_op.nfe.zero_()
        return odeint(self.transport_op, z0_det, t_span,
                    method='rk4',
                    options={'step_size': dt})

    def compute_velocities(self, z: torch.Tensor,) -> torch.Tensor:
        return self.transport_op.compute_velocities(z)

    def flag_direction_loss_graph_global(
        self,
        z0: torch.Tensor,           # (B, D) 
        v0: torch.Tensor,           # (B, D)
        is_start: torch.BoolTensor, # (B,)
        is_term:  torch.BoolTensor, # (B,)
        z_all:    torch.Tensor,     # (N, D)  global cache
        nbr_idx_global: torch.LongTensor,  # (N, K)  global kNN
        idx_global: torch.LongTensor,      # (B,)
        nbr_mask_global: torch.BoolTensor = None,  # (N, K) optional
    ) -> torch.Tensor:
        nbrs = nbr_idx_global[idx_global]       # (B, K) global indices
        z_nbrs = z_all[nbrs]                     # (B, K, D)
        
        if nbr_mask_global is not None:
            mask_f = nbr_mask_global[idx_global].unsqueeze(-1).float()
            mean_n = (z_nbrs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            mean_n = z_nbrs.mean(dim=1)
        
        projs = (v0 * (mean_n - z0)).sum(dim=1)
        losses = []
        if is_start.any():
            losses.append(F.relu(-projs[is_start]).mean())
        if is_term.any():
            losses.append(F.relu(projs[is_term]).mean())
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=z0.device)    