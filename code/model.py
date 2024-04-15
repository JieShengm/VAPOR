import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.neighbors import NearestNeighbors
from utilities import plot_pairs

class VAE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 latent_dim, 
                 psi_M,
                 encoder_dims=[512, 256], 
                 decoder_dims=[256, 512]):
        
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.psi_M = psi_M
        
        encoder_layers = []
        for in_dim, out_dim in zip([input_dim] + encoder_dims[:-1], encoder_dims):
            encoder_layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)
        self.convert_mu_nbrs = nn.Linear(self.latent_dim*self.psi_M, latent_dim, bias=False)
        
        decoder_layers = []
        for in_dim, out_dim in zip([latent_dim] + decoder_dims[:-1], decoder_dims):
            decoder_layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        decoder_layers.append(nn.Linear(decoder_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def update_linear_weights_according_to_psi(self, psi_filtered_indices):
        if psi_filtered_indices is not None:
            complete_mask = torch.zeros(self.psi_M, dtype=torch.bool)
            complete_mask[psi_filtered_indices] = True
            expanded_mask = complete_mask.repeat_interleave(repeats=self.latent_dim)
            original_weights = self.convert_mu_nbrs.weight.detach().clone()
            new_weights = original_weights[:, expanded_mask]
            self.convert_mu_nbrs.weight = nn.Parameter(new_weights)
        else:
            print('No indices provided for filtering.')

    def Encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def Decode(self, z):
        return self.decoder(z)

    def forward(self, x, mu_nbrs=None,psi_filtered_indices=None):
        mu, logvar = self.Encode(x)
        if mu_nbrs is not None:
            m, batch_size, latent_dim = mu_nbrs.shape
            mu_nbrs = mu_nbrs.permute(1, 0, 2).reshape(batch_size, latent_dim * m)
            if psi_filtered_indices is not None:
                self.update_linear_weights_according_to_psi(psi_filtered_indices)
            mu_ast = self.convert_mu_nbrs(mu_nbrs)
        else:
            mu_ast = mu
        z = self.reparameterize(mu_ast, logvar)
        reconstructed = self.Decode(z)
        return reconstructed, z, mu_ast, logvar  

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
        self.psi = torch.empty([latent_dim, latent_dim, M]).normal_(mean=0,std=0.1).to(device)
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
        m = self.psi.shape[2]
        
        c = torch.zeros([batch_size, 1, m]).to(device).requires_grad_(True)
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
        trans_op = torch.matrix_exp(torch.sum(torch.einsum('bim,jkm->bjkm', c, psi), dim=-1))
        recon_loss = (((z1-torch.einsum('bij,bj->bi', trans_op, z0))**2).sum())/batch_size
        trans_op_loss = (torch.norm(psi, p='fro',dim=[0,1])**2).sum()
        coef_loss = torch.abs(c).sum()
        

        energy = recon_loss + self.gamma * trans_op_loss + self.zeta * coef_loss
        if return_all:
            return energy, recon_loss, trans_op_loss, coef_loss
        else: 
            return energy

    def filter_psi(self, psi, c, epsilon=1e-8):
        norms = torch.norm(psi, p='fro',dim=[0,1])**2
        sorted_indices = torch.argsort(norms,descending=True)
        filtered_indices = sorted_indices[norms[sorted_indices] > epsilon]
        m = psi.shape[2]
        m_filtered = filtered_indices.shape[0]
        if m_filtered==0 or m_filtered==m:
            psi = psi[:,:,sorted_indices]
            c = c[:,:,sorted_indices]
        else:
            psi = psi[:,:,filtered_indices]
            c = c[:,:,filtered_indices]
            print("Filtered M, M = ",m_filtered)
        return psi, c, filtered_indices

def construct_pairs(x, n_neighbors=15, psi = None):
    # Compute nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(x.cpu().numpy())
    distances, indices = nbrs.kneighbors(x.cpu().numpy())
    indices = torch.tensor(indices).to(x.device)
    distances = torch.tensor(distances).float().to(x.device)

    # Basic probabilities based on distances
    rho = torch.min(distances[:, 1:], dim=1).values
    sigma = torch.std(distances[:, 1:], dim=1)
    probabilities = torch.exp(-(distances[:, 1:] - rho.unsqueeze(1)) / sigma.unsqueeze(1))
    probabilities /= probabilities.sum(dim=1, keepdim=True)
    # print('probabilities before: ', probabilities[0])

    from torch.nn.functional import normalize
    x1 = torch.einsum('mik, bk -> mbi', torch.matrix_exp(psi.permute(2, 0, 1)), x)
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