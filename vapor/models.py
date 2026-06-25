import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from .utils import init_vae_weights

# ============================================================
# Transport Operator
# ============================================================

class TransportOperator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_dynamics: int,
        gate_temperature: float = 0.5,
        gate_mode: str = "sigmoid_norm",
        use_bias: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_dynamics = n_dynamics
        self.gate_mode = gate_mode
        self.use_bias = use_bias

        valid_modes = {"softmax", "sigmoid_norm", "sigmoid"}
        if gate_mode not in valid_modes:
            raise ValueError(
                f"Unknown gate_mode: '{gate_mode}'. Must be one of {sorted(valid_modes)}"
            )

        # (M, D, D) — single batched tensor instead of ParameterList
        self.Psi = nn.Parameter(torch.empty(n_dynamics, latent_dim, latent_dim))
        for m in range(n_dynamics):
            nn.init.orthogonal_(self.Psi.data[m], gain=1.0)

        if use_bias:
            self.b = nn.Parameter(torch.zeros(n_dynamics, latent_dim))
            nn.init.normal_(self.b, mean=0.0, std=0.01)
        else:
            self.register_parameter("b", None)

        self.gate_tokens = nn.Parameter(torch.randn(n_dynamics, latent_dim))
        nn.init.xavier_uniform_(self.gate_tokens)

        self.register_buffer("gate_temperature", torch.tensor(float(gate_temperature)))
        self.eps = 1e-8

        self.speed_head = nn.Sequential(
            nn.Linear(latent_dim, max(32, latent_dim // 4)),
            nn.LeakyReLU(),
            nn.Linear(max(32, latent_dim // 4), 1),
            nn.Softplus(),
        )
    
    def raw_directions(self, z: torch.Tensor) -> torch.Tensor:
        U = torch.einsum('bd,mde->bme', z, self.Psi)                 # (B, M, D)
        if self.use_bias:
            U = U + self.b.unsqueeze(0)
        return U

    def unit_directions(self, z: torch.Tensor) -> torch.Tensor:
        U = self.raw_directions(z)                                    # (B, M, D)
        return U / (U.norm(dim=-1, keepdim=True).clamp_min(1e-6))     # (B, M, D)

    def gate_logits(self, z: torch.Tensor, Uhat: torch.Tensor = None) -> torch.Tensor:
        if Uhat is None:
            Uhat = self.unit_directions(z)
        tok = F.normalize(self.gate_tokens, dim=-1, eps=1e-6)         # (M, D)
        logits = torch.einsum('bmd,md->bm', Uhat, tok)               # (B, M)
        return logits / self.gate_temperature.clamp_min(1e-6)

    def get_mixture_weights(self, z: torch.Tensor, Uhat: torch.Tensor = None) -> torch.Tensor:
        logits = self.gate_logits(z, Uhat=Uhat)                       # (B, M)
        mode = self.gate_mode.lower()
        if mode == "softmax":
            pi = torch.softmax(logits, dim=1)
        elif mode == "sigmoid_norm":
            a = torch.sigmoid(logits)
            pi = a / (a.sum(dim=1, keepdim=True) + self.eps)
        elif mode == "sigmoid":
            pi = torch.sigmoid(logits)
        else:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")
        return pi
    
    # ---------------------------------------------------------------------
    # Forward / velocity
    # ---------------------------------------------------------------------
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        Uhat = self.unit_directions(z)                                # (B, M, D)
        pi = self.get_mixture_weights(z, Uhat=Uhat)                   # (B, M)
        v_dir = torch.einsum('bm,bmd->bd', pi, Uhat)                  # (B, D)
        speed = self.speed_head(z).squeeze(-1)                         # (B,)
        return v_dir * speed.unsqueeze(-1)                            # (B, D)
    
    @torch.no_grad()
    def unit_velocity(self, z: torch.Tensor) -> torch.Tensor:
        Uhat = self.unit_directions(z)
        pi = self.get_mixture_weights(z, Uhat=Uhat)
        v = torch.einsum('bm,bmd->bd', pi, Uhat)
        return F.normalize(v, dim=-1, eps=1e-6)
    
    def compute_velocities(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(torch.tensor(0.0, device=z.device), z)
    
    def orthogonality_loss(self, mode: str = "stiefel") -> torch.Tensor:
        """Penalize the M Psi operators for not being orthogonal.

        mode='stiefel': ||P^T P - I||_F^2
            Constrains BOTH direction (off-diagonals -> 0, i.e. Psi_m
            mutually orthogonal) AND norm (diagonals -> 1, i.e. each
            ||Psi_m|| -> 1). Prevents collapse, but assumes the GT
            operators have comparable Frobenius norms — biased when one
            process is dominated by an affine bias and its linear Psi
            is near zero.

        mode='cosine': ||off_diag( P_n^T P_n )||_F^2  with P_n column-normalized
            Only constrains direction (cosines between unit-vectorized
            Psi_m -> 0). Norms are free, so processes with very different
            magnitudes (e.g. strong rotation vs weak linear-drift +
            large affine bias) are not distorted. Does NOT prevent
            collapse on its own — pair with a small weight decay on Psi
            or a norm hinge if you observe ||Psi_m|| -> 0.
        """
        P = self.Psi.reshape(self.n_dynamics, -1).T                    # (d^2, M)
        M_ = P.shape[1]
        eye_M = torch.eye(M_, device=P.device, dtype=P.dtype)
        if mode == "stiefel":
            gram = P.T @ P
            return ((gram - eye_M) ** 2).sum()
        elif mode == "cosine":
            P_n = P / (P.norm(dim=0, keepdim=True) + 1e-9)
            gram = P_n.T @ P_n
            off = gram - eye_M
            return (off ** 2).sum()
        else:
            raise ValueError(f"unknown orthogonality mode: {mode}")

# ============================================================
# VAE
# ============================================================

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, encoder_dims=None, decoder_dims=None):
        super().__init__()
        encoder_dims = encoder_dims or [512, 256, 128]
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
    
    def integrate(self, z0: torch.Tensor, t_span: torch.Tensor,
              dt: float = 1.0):
        z0_det = z0.detach()
        if hasattr(self.transport_op, "nfe"):
            self.transport_op.nfe.zero_()
        return odeint(self.transport_op, z0_det, t_span,
                    method='rk4',
                    options={'step_size': dt})

    def compute_velocities(self, z: torch.Tensor) -> torch.Tensor:
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