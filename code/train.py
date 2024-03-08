import torch
from model import construct_pairs, vae_to_loss
import time

def train_transport_operator(train_loader, vae, transport_operator, device, args):
    total_energy, total_recon_loss, total_trans_op_loss, total_coef_loss = 0, 0, 0, 0
    for mini_batch, data in enumerate(train_loader):
        data = data.float().to(device)
        with torch.no_grad():
            z0, _, _ = vae.Encode(data)
        pairs = construct_pairs(z0)

        start_time = time.time()
        psi, c = transport_operator.E_step(pairs, max_iterations=args.max_iterations)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"mini batch{mini_batch}: E_step completed in {elapsed_time:.2f} seconds.")
        
        start_time = time.time()
        psi, c = transport_operator.M_step(pairs, psi, c, max_iterations=args.max_iterations)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"mini batch{mini_batch}: M_step completed in {elapsed_time:.2f} seconds.")
        
        # Compute norms and losses
        psi_norm_squared = torch.norm(psi, p='fro', dim=[0, 1])**2
        for i in range(psi_norm_squared.size(0)):
            print(f"psi {i} norm: {psi_norm_squared[i].item():.12f}")
        
        energy, recon_loss, trans_op_loss, coef_loss = transport_operator.energy_function(pairs, psi, c, return_all=True)
        total_energy += energy
        total_recon_loss += recon_loss
        total_trans_op_loss += trans_op_loss
        total_coef_loss += coef_loss
    return total_energy, total_recon_loss, total_trans_op_loss, total_coef_loss

def train_vae(train_loader, vae, transport_operator, optimizer_vae, train_vaeto, device, args):
    train_loss, total_bce, total_kld, total_to_transformed_mse = 0, 0, 0, 0
    MSE = None
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
            z0, _, _ = vae.Encode(data)
            pairs = construct_pairs(z0)
            psi, c = transport_operator.E_step(pairs, max_iterations=args.max_iterations)
            z0_ast = vae.transform_trans_op(pairs, psi, c)
            recon, _, mu, logvar = vae(data, z0_ast=z0_ast)
            BCE, KLD, MSE = vae_to_loss(data, recon, mu, logvar, z0_ast, z0)
            loss = BCE + KLD

        loss.backward()
        train_loss += loss.item()
        total_bce += BCE.item()
        total_kld += KLD.item()
        total_to_transformed_mse += MSE.item() if MSE is not None else 0
        optimizer_vae.step()
    return train_loss, total_bce, total_kld, total_to_transformed_mse
