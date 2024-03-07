import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.neighbors import NearestNeighbors

class VAE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 latent_dim, 
                 encoder_dims=[512, 256], 
                 decoder_dims=[256, 512]):
        
        super(VAE, self).__init__()
        
        encoder_layers = []
        for in_dim, out_dim in zip([input_dim] + encoder_dims[:-1], encoder_dims):
            encoder_layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)
        
        decoder_layers = []
        for in_dim, out_dim in zip([latent_dim] + decoder_dims[:-1], decoder_dims):
            decoder_layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        decoder_layers.append(nn.Linear(decoder_dims[-1], input_dim))
        decoder_layers.append(nn.LeakyReLU())  # Adjust the final activation as needed
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def Encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z0 = self.reparameterize(mu, logvar)
        return z0, mu, logvar
    
    def Decode(self, z):
        return self.decoder(z)

    def forward(self, x, z0_ast=None):
        z0, mu, logvar = self.Encode(x)
        z = z0_ast if z0_ast is not None else z0
        reconstructed = self.Decode(z)
        return reconstructed, z0, mu, logvar
        
    def transform_trans_op(self, pairs, psi, c):
        z0, z1 = pairs
        device=z0.device

        trans_op = torch.matrix_exp(torch.sum(torch.einsum('bim,jkm->bjkm', c, psi), dim=-1))
        batch_size, latent_dim, _ = trans_op.shape
        z0_ast = torch.zeros_like(z1)

        dets = torch.linalg.det(trans_op)
        for b in range(batch_size):
            if dets[b] != 0:
                trans_op_inv = torch.linalg.inv(trans_op[b])
                z0_ast[b] = torch.einsum('ij,j->i', trans_op_inv, z1[b]) + 0.001 * torch.randn_like(z1[b], device=device)
            else:
                print(f"Matrix at index {b} is not invertible.")
                return None
        return z0_ast

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

       self.trans_op_training_counter = 0

    def E_step(self, 
               pairs, 
               stopping_criteria='absolute', 
               threshold=1e-8,
               min_iterations = 200, 
               max_iterations= 500):
        
        z0, z1 = pairs
        device = z0.device
        batch_size = z0.shape[0]
        m = self.psi.shape[2]
        
        c = torch.zeros([batch_size, 1, m], device=device).requires_grad_(True)
        psi = self.psi

        optimizer_c = optim.AdamW([c], self.lr_eta_E)
        prev_loss = float('inf')

        for i in range(min_iterations, max_iterations):
            optimizer_c.zero_grad()
            loss = self.energy_function(pairs, psi, c)
            loss.backward()
            optimizer_c.step()

            current_loss = loss.item()
            if stopping_criteria == 'absolute':
                diff = abs(prev_loss - current_loss)
            elif stopping_criteria == 'relative':
                diff = abs(prev_loss - current_loss) / (abs(prev_loss) + 1e-8)

            if diff < threshold:
                # print(f"E-step: Convergence reached at iteration {i+1} with loss: {current_loss:.6f}")
                # print(f'E-step: diff: {diff: .12f}; prev_loss: {prev_loss: .6f}; current_loss: {current_loss: .6f}')
                break
            # elif (i+1) == max_iterations:
            #     print(f"E-step: Stop at iteration {i+1} with loss: {current_loss:.6f}")

            prev_loss = current_loss

        c = c.detach()
        self.c = c
        self.psi = psi

        return psi, c
    
    def M_step(self, pairs, psi, c,
               stopping_criteria='absolute', 
               initial_threshold = 1e-4,
               decay_rate = 0.95,
               max_iterations = 10000):
         
        z0, z1 = pairs

        psi = psi.requires_grad_(True)

        optimizer_psi = optim.AdamW([psi], self.lr_eta_M)
        prev_loss = float('inf')
        threshold = initial_threshold * (decay_rate ** self.trans_op_training_counter)
        
        for i in range(max_iterations):
            optimizer_psi.zero_grad()
            loss = self.energy_function(pairs, psi, c)
            loss.backward()
            optimizer_psi.step()
            
            current_loss = loss.item()
            
            if stopping_criteria == 'absolute':
                diff = abs(prev_loss - current_loss)
            elif stopping_criteria == 'relative':
                diff = abs(prev_loss - current_loss) / (abs(prev_loss) + 1e-8)

            if diff < threshold:
                print(f"M-step: Convergence reached at iteration {i+1} with loss: {current_loss:.6f}")
                print(f'M-step: diff: {diff: .12f}; prev_loss: {prev_loss: .6f}; current_loss: {current_loss: .6f}')
                break
            elif (i+1) == max_iterations:
                print(f"M-step: Stop at iteration {i+1} with loss: {current_loss:.6f}")

            prev_loss = current_loss

        threshold *= decay_rate

        self.psi, self.c = self.filter_psi(psi.detach(),c)
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

    def filter_psi(self, psi, c, epsilon=1e-6):
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
        return psi, c
    
    def update_to_training_counter(self):
        self.trans_op_training_counter += 1

def construct_pairs(z0, n_neighbors=50):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(z0.cpu())
    distances, indices = nbrs.kneighbors(z0.cpu())
    indices = torch.tensor(indices).to(z0.device) 
    distances = torch.tensor(distances).to(z0.device)

    rho = torch.min(distances[:, 1:], dim=1).values
    sigma = torch.std(distances[:, 1:], dim=1)
    probabilities = torch.exp((distances[:, 1:] - rho.unsqueeze(1)) / sigma.unsqueeze(1))
    probabilities /= probabilities.sum(dim=1, keepdim=True)
        
    chosen_neighbors = torch.multinomial(probabilities, 1).squeeze()
    chosen_indices = indices[torch.arange(len(z0)), chosen_neighbors + 1]
    pairs = [z0, z0[chosen_indices]]
    return pairs

def vae_to_loss(x, recon_x, mu, logvar,  z0_ast=None, z0 = None):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if z0_ast is not None:
        MSE = nn.functional.mse_loss(z0_ast, z0, reduction='sum')
        return BCE, KLD, MSE
    else: 
        return BCE, KLD