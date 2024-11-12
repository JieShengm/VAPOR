import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.neighbors import NearestNeighbors
# from utilities import plot_pairs

class VAE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 latent_dim, 
                 psi_M,
                 encoder_dims=[512, 256], 
                 decoder_dims=[256, 512],):
        
        super(VAE, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.latent_dim = latent_dim
        self.psi_M = psi_M
        
        encoder_layers = []
        for in_dim, out_dim in zip([input_dim] + encoder_dims[:-1], encoder_dims):
            encoder_layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)
        # self.convert_mu_nbrs = nn.Linear(self.latent_dim*self.psi_M, latent_dim, bias=False)
        
        decoder_layers = []
        for in_dim, out_dim in zip([latent_dim] + decoder_dims[:-1], decoder_dims):
            decoder_layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        decoder_layers.append(nn.Linear(decoder_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
            
        self.register_buffer('psi_mask', torch.ones(psi_M))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def update_psi_mask(self, filtered_indices):
        if filtered_indices is not None:
            # self.psi_M = len(filtered_indices)
            self.psi_mask = torch.zeros(self.psi_M, device = self.device)
            self.psi_mask[filtered_indices] = 1
            self.gate_indices = filtered_indices  # Store indices for all variants
        else:
            print('No indices provided for filtering.')

    def aggregate_neighbor_states(self, mu_nbrs, psi):
        # """Add size checking prints"""
        # print("\nAggregation Sizes:")
        # print(f"mu_nbrs shape: {mu_nbrs.shape}")
        # print(f"psi shape: {psi.shape}")
        # print(f"psi_mask shape: {self.psi_mask.shape}")
        
        # masked_psi = psi * self.psi_mask.view(1, 1, -1)
        # print(f"masked_psi shape: {masked_psi.shape}")
        
        # transformed = torch.einsum('ijk,bkm->bij', masked_psi, mu_nbrs.permute(1, 0, 2))
        # print(f"transformed shape: {transformed.shape}")
        raise NotImplementedError("Implement in subclass")
    
    # def update_linear_weights_according_to_psi(self, psi_filtered_indices):
    #     if psi_filtered_indices is not None:
    #         original_weights = self.convert_mu_nbrs.weight.detach().clone()
    #         M = original_weights.shape[1]//self.latent_dim
    #         complete_mask = torch.zeros(M, dtype=torch.bool)
    #         complete_mask[psi_filtered_indices] = True
    #         expanded_mask = complete_mask.repeat_interleave(repeats=self.latent_dim)
    #         new_weights = original_weights[:, expanded_mask]
    #         self.convert_mu_nbrs.weight = nn.Parameter(new_weights)
    #     else:
    #         print('No indices provided for filtering.')

    def Encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def Decode(self, z):
        return self.decoder(z)

    def forward(self, x, mu_nbrs=None, psi=None, psi_filtered_indices=None):
        mu, logvar = self.Encode(x)
        
        if mu_nbrs is not None and psi is not None:
            print("\nInside VAE forward, incorporating mu_ast:")
            # print(f"mu_nbrs shape: {mu_nbrs.shape}") #[M, batch_size, latent_dim]
            # print(f"psi shape: {psi.shape}") #[M, latent_dim, latent_dim]
            # print(f"psi_filtered_indices: {psi_filtered_indices}") # [M]
            # Update psi_mask to match new psi size
            # if psi_filtered_indices is not None:
            self.update_psi_mask(psi_filtered_indices)
            # Slice mu_nbrs to match new size
            mu_nbrs = mu_nbrs[:len(psi_filtered_indices)]
            mu_ast = self.aggregate_neighbor_states(mu_nbrs, psi)
        else:
            # print("\nInside VAE forward, training on mu:")
            mu_ast = mu
            
        z = self.reparameterize(mu_ast, logvar)
        reconstructed = self.Decode(z)
        return reconstructed, z, mu_ast, logvar  

class DirectSumVAE(VAE):
    def aggregate_neighbor_states(self, mu_nbrs, psi):
        """Simple summation over transformed states
        Args:
            mu_nbrs: [M, batch_size, latent_dim]
            psi: [M, latent_dim, latent_dim]
        """
        transformed = torch.einsum('mij,bmj->bmi', psi, mu_nbrs.permute(1, 0, 2))
        return torch.sum(transformed, dim=1)

class WeightedSumVAE(VAE):
    def __init__(self, input_dim, latent_dim, psi_M, **kwargs):
        super().__init__(input_dim, latent_dim, psi_M, **kwargs)
        self.aggregation_weights = nn.Parameter(torch.ones(psi_M))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Register the backward hook
        self.aggregation_weights.register_hook(self._aggregation_weights_hook)

    def _aggregation_weights_hook(self, grad):
        # Zero out gradients where psi_mask == 0
        return grad * self.psi_mask
    
    def aggregate_neighbor_states(self, mu_nbrs, psi):
        """
        Args:
            mu_nbrs: [M, batch_size, latent_dim]
            psi: [M, latent_dim, latent_dim]
        """
        print("\nWeightedSumVAE Dimension Check:")
        print(f"1. Input shapes:")
        print(f"   mu_nbrs: {mu_nbrs.shape}")  # [4, 512, 3]
        print(f"   psi: {psi.shape}")          # [4, 3, 3]
        print(f"   psi_mask: {self.psi_mask.shape}")  # [4]
        
        # First normalize weights for active psi matrices
        # [4] -> softmax -> [4]
        weights = F.softmax(self.aggregation_weights / self.temperature, dim=0)
        print(f"\n2. Normalized weights: {weights}")
        
        # Apply mask to weights
        masked_weights = weights * self.psi_mask
        # Re-normalize after masking
        masked_weights = masked_weights / (masked_weights.sum() + 1e-8)
        print(f"3. Masked weights: {masked_weights}")
        
        # Transform each neighbor with each psi
        transformed = torch.einsum('mij,bmj->bmi', psi, mu_nbrs.permute(1, 0, 2))
        print(f"\n4. Transformed shape: {transformed.shape}")  # [512, 4, 3]
        
        # Apply weights: [512, 4, 3] * [4] -> [512, 3, 4]
        weighted = transformed * masked_weights.view(1, self.psi_mask.shape[0], 1)
        
        # Sum over psi matrices
        output = torch.sum(weighted, dim=-1)  # [512, 3]
        
        return output

class GatedSumVAE(VAE):
    def __init__(self, input_dim, latent_dim, psi_M, **kwargs):
        super().__init__(input_dim, latent_dim, psi_M, **kwargs)
        self.gate_network = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, psi_M),
            nn.Sigmoid()
        )

    def aggregate_neighbor_states(self, mu_nbrs, psi):
        """Dynamic gating based on input states
        Args:
            mu_nbrs: [M, batch_size, latent_dim]
            psi: [M, latent_dim, latent_dim]
        """
        # Transform: [batch_size, M, latent_dim]
        transformed = torch.einsum('mij,bmj->bmi', psi, mu_nbrs.permute(1, 0, 2))
        
        # Compute dynamic gates
        gates = self.gate_network(mu_nbrs.mean(dim=0))  # [batch_size, M]
        
        # Apply mask to gates
        gates = gates * self.psi_mask.view(1, -1)  # [batch_size, psi_M]
        
        # Re-normalize gates to sum to 1
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)
        
        # Aggregate transformed neighbor states using gates
        output = torch.sum(transformed * gates.unsqueeze(-1), dim=1)  # [batch_size, latent_dim]
        
        return output


class TransportOperator(nn.Module):
    def __init__(self, 
                 latent_dim,
                 M,
                 gamma,
                 zeta,
                 lr_eta_E,
                 lr_eta_M):
        super(TransportOperator, self).__init__() 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## change this later
        self.psi = torch.empty([M, latent_dim, latent_dim]).normal_(mean=0,std=0.1).to(device)
        self.c = None
        
        self.gamma = gamma
        self.zeta = zeta
        self.lr_eta_E = lr_eta_E
        self.lr_eta_M = lr_eta_M 
        
        self.filtered_indices = None

    def E_step(self, 
               pairs, 
               max_iterations = None,):
        pairs = (pairs[0].detach(), pairs[1].detach())

        device = pairs[0].device
        batch_size = pairs[0].shape[0]
        m = self.psi.shape[0]
        
        c = torch.zeros([batch_size, m, 1]).to(device).requires_grad_(True)
        psi = self.psi
        optimizer_c = optim.AdamW([c], self.lr_eta_E)
        
        for i in range(max_iterations):
            optimizer_c.zero_grad()
            loss = self.energy_function(pairs, psi, c)
            loss.backward()
            optimizer_c.step()

        c = c.detach()
        self.c = c
        self.psi = psi

        return psi, c
    
    def M_step(self, pairs, psi, c,max_iterations = None,): 
        pairs = (pairs[0].detach(), pairs[1].detach())

        psi = psi.requires_grad_(True)
        optimizer_psi = optim.AdamW([psi], self.lr_eta_M)
        
        for i in range(max_iterations):
            optimizer_psi.zero_grad()
            loss = self.energy_function(pairs, psi, c)
            loss.backward()
            optimizer_psi.step()

        self.psi, self.c, self.filtered_indices = self.filter_psi(psi.detach(),c)
        return self.psi, self.c

    def energy_function(self, pairs, psi, c, return_all = False):
        z0, z1 = pairs
        batch_size = z0.shape[0]
        trans_op = torch.matrix_exp(torch.sum(torch.einsum('bmi,mjk->bmjk', c, psi), dim =1)) 
        recon_loss = (((z1-torch.einsum('bij,bj->bi', trans_op, z0))**2).sum())/batch_size
        trans_op_loss = (torch.norm(psi, p='fro',dim=[1,2])**2).sum()
        coef_loss = torch.abs(c).sum()
        

        energy = recon_loss + self.gamma * trans_op_loss + self.zeta * coef_loss
        if return_all:
            return energy, recon_loss, trans_op_loss, coef_loss
        else: 
            return energy

    def filter_psi(self, psi, c, epsilon=1e-8):
        norms = torch.norm(psi, p='fro',dim=[1,2])**2
        sorted_indices = torch.argsort(norms,descending=True)
        filtered_indices = sorted_indices[norms[sorted_indices] > epsilon]
        m = psi.shape[0]
        m_filtered = filtered_indices.shape[0]
        if m_filtered==0 or m_filtered==m:
            print("No filtering performed (all kept or all removed)")
            psi = psi[sorted_indices]
            c = c[:,sorted_indices,:]
        else:
            print(f"Filtering performed: {m} -> {m_filtered}")
            psi = psi[filtered_indices]
            c = c[:,filtered_indices,:]
        return psi, c, filtered_indices

def construct_pairs(x, n_neighbors=30, psi = None):
    # Compute nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(x.cpu().numpy())
    distances, indices = nbrs.kneighbors(x.cpu().numpy())
    indices = torch.tensor(indices).to(x.device)
    distances = torch.tensor(distances).float().to(x.device)

    # Basic probabilities based on distances
    rho = torch.min(distances[:, 1:], dim=1).values
    sigma = torch.std(distances[:, 1:], dim=1)
    probabilities = torch.exp((distances[:, 1:] - rho.unsqueeze(1)) / sigma.unsqueeze(1))
    probabilities /= probabilities.sum(dim=1, keepdim=True)
    # print('probabilities before: ', probabilities[0])

    from torch.nn.functional import normalize
    x1 = torch.einsum('mik, bk -> mbi', torch.matrix_exp(psi), x)
    v_normalized = normalize(x1 - x, p=2, dim=-1)
    nbrs_indices = indices[:, 1:]
    direction_vectors = x[nbrs_indices] - x.unsqueeze(1)
    direction_vectors_normalized = normalize(direction_vectors, p=2, dim=-1) 
    cos_sim = cos_sim = torch.einsum('ijk,mik->mij', direction_vectors_normalized, v_normalized)
    epsilon = 1e-5
    cos_sim = (cos_sim - cos_sim.min(dim=2, keepdim=True)[0] + epsilon) / \
                (cos_sim.max(dim=2, keepdim=True)[0] - cos_sim.min(dim=2, keepdim=True)[0] + epsilon)  
                
    # Average cosine similarity for first neighbor selection
    mean_cos_sim = cos_sim.mean(dim=0)         
    adjusted_probabilities_mean = probabilities * mean_cos_sim
    adjusted_probabilities_mean /= adjusted_probabilities_mean.sum(dim=1, keepdim=True)
    chosen_indices = []
    chosen_index = torch.multinomial(adjusted_probabilities_mean, 1).squeeze()
    chosen_indices.append(indices[torch.arange(len(x)), chosen_index + 1])

    # Adjust probabilities for each neighbor based on its cosine similarity
    adjusted_probabilities = probabilities * cos_sim
    for i in range(adjusted_probabilities.shape[0]):
        adj_probs = adjusted_probabilities[i]
        adj_probs /= adj_probs.sum(dim=1, keepdim=True)
        chosen_index = torch.multinomial(adj_probs, 1).squeeze()
        chosen_indices.append(indices[torch.arange(len(x)), chosen_index + 1])
        
    chosen_indices = torch.stack(chosen_indices).to(x.device)
    nbr_pairs = [x,torch.stack([x[chosen_indices[i]] for i in range(chosen_indices.shape[0])])]
    return nbr_pairs 

def vae_to_loss(x, recon_x, mu, logvar):
    batch_size = x.shape[0]
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')/batch_size
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/batch_size
    return BCE, KLD
