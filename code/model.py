import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors

class VAE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 latent_dim, 
                 encoder_dims=[512, 256], 
                 decoder_dims=[256, 512]):
        
        super(VAE, self).__init__()
        
        # Create the encoder layers
        encoder_layers = []
        for in_dim, out_dim in zip([input_dim] + encoder_dims[:-1], encoder_dims):
            encoder_layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)
        
        # Create the decoder layers
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

    def forward(self, x, return_latent = False):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z0 = self.reparameterize(mu, logvar)
        if return_latent:
            return z0
        else:
            return self.decoder(z0), mu, logvar

class TransportOperator(nn.Module):
    def __init__(self, 
                 #z0,
                 latent_dim,
                 M):
       
       super(TransportOperator, self).__init__() 

       #Initialization
       print('it should run only once')
       self.psi = torch.empty([latent_dim, latent_dim, M]).normal_(mean=0,std=0.1)
       print(f"psi shape: {self.psi.shape} (run only once)")

    def construct_pairs(self, z0, n_neighbors=50):
        # Fit nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(z0.cpu())
        distances, indices = nbrs.kneighbors(z0.cpu())
        distances = torch.tensor(distances).to(z0.device)

        # print(distances) # ensure the first column is zeros: YES
        rho = torch.min(distances[:, 1:], dim=1).values
        sigma = torch.std(distances[:, 1:], dim=1)
        probabilities = torch.exp((distances[:, 1:] - rho.unsqueeze(1)) / sigma.unsqueeze(1))
        probabilities /= probabilities.sum(dim=1, keepdim=True)
        
        chosen_neighbors = torch.multinomial(probabilities, 1).squeeze()
        chosen_indices = indices[torch.arange(len(z0)), chosen_neighbors + 1]
        pairs = [z0, z0[chosen_indices]]
        return pairs

    def filter_psi(self, c_min, epsilon=1e-6):
        norms = torch.norm(self.psi, p='fro',dim=[0,1])**2
        sorted_indices = torch.argsort(norms,descending=True)
        filtered_indices = sorted_indices[norms[sorted_indices] > epsilon]
        m = self.psi.shape[2]
        m_filtered = filtered_indices.shape[0]
        if m_filtered==0 or m_filtered==m:
            self.psi = self.psi[:,:,sorted_indices]
            self.c = c_min[:,:,sorted_indices]
        else:
            self.psi = self.psi[:,:,filtered_indices]
            self.c = c_min[:,:,filtered_indices]
            print("Filtered M, M = ",m_filtered)
        return
    


    pass
       
def vae_loss(x, recon_x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD