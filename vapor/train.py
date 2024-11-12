import torch
from .model import construct_pairs, vae_to_loss
import time

def train_transport_operator(train_loader, vae, transport_operator, train_vaeto, device, config):
    total_energy, total_recon_loss, total_trans_op_loss, total_coef_loss = 0, 0, 0, 0
    
    print("--TO model--")
    if not train_vaeto:
        max_iterations = max(10, config['max_iterations'])
        # to_learning_n_minibatch = float('inf')
        # print('--TO model--: VAE init phase')
    else:
        max_iterations = max(10, config['max_iterations'])
        # to_learning_n_minibatch = args.to_learning_n_minibatch
        # print('--TO model--: VAE+TO phase')

    for mini_batch, data in enumerate(train_loader):
        data = data.float().to(device)
        with torch.no_grad():
            mu, logvar = vae.Encode(data)
        pairs = construct_pairs(mu.detach(), psi = transport_operator.psi.detach())
        pairs = [pairs[0],pairs[1][0]]

        start_time = time.time()
        psi, c = transport_operator.E_step(pairs, max_iterations=max_iterations)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"mini batch{mini_batch}: E_step completed in {elapsed_time:.2f} seconds. (max_iterations = {max_iterations})")
        
        start_time = time.time()
        psi, c = transport_operator.M_step(pairs, psi, c, max_iterations=max_iterations)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"mini batch{mini_batch}: M_step completed in {elapsed_time:.2f} seconds. (max_iterations = {max_iterations})")
        
        # Compute norms and losses
        psi_norm_squared = torch.norm(psi, p='fro', dim=[1, 2])**2
        for i in range(psi_norm_squared.size(0)):
            print(f"psi {i} norm: {psi_norm_squared[i].item():.12f}")
        
        energy, recon_loss, trans_op_loss, coef_loss = transport_operator.energy_function(pairs, psi, c, return_all=True)
        total_energy += energy
        total_recon_loss += recon_loss
        total_trans_op_loss += trans_op_loss
        total_coef_loss += coef_loss
    return total_energy, total_recon_loss, total_trans_op_loss, total_coef_loss

def train_vae(train_loader, vae, transport_operator, optimizer_vae, train_vaeto, device, config):
    train_loss, total_bce, total_kld = 0, 0, 0
            
    for _, data in enumerate(train_loader):
        data = data.float().to(device)
        optimizer_vae.zero_grad()
        if not train_vaeto:
            # Warmup phase: Train as a conventional VAE
            recon, _, mu, logvar = vae(data)
            BCE, KLD = vae_to_loss(data, recon, mu, logvar)
            loss = BCE + KLD
        else:
            # Train VAE with TO integration after warmup
            mu, _ = vae.Encode(data)
            pairs = construct_pairs(mu.detach(), psi = transport_operator.psi.detach())
            recon, _, mu_ast, logvar = vae(
                data, 
                mu_nbrs=pairs[1][1:],  # neighbor encodings
                psi=transport_operator.psi,  # current psi
                psi_filtered_indices=transport_operator.filtered_indices  # filtered indices
            )
            BCE, KLD = vae_to_loss(data, recon, mu_ast, logvar)
            loss = BCE + KLD

        loss.backward()
        train_loss += loss.item()
        total_bce += BCE.item()
        total_kld += KLD.item()
        optimizer_vae.step()
    return train_loss, total_bce, total_kld
