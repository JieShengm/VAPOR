import argparse
import os
import datetime
from pathlib import Path

import torch
import torch.optim as optim

from model_mu import VAE, TransportOperator
from train_mu import train_vae, train_transport_operator
from utilities import get_dataloader, load_checkpoint


def get_args_parser():
    parser = argparse.ArgumentParser(description="Train VAE.")
    
    parser.add_argument("--data_path", type=str, default='./data/Pasca/val_pasca_subset_expr.csv', help="Data Path.")
    parser.add_argument("--checkpoint_name", type=str, default='checkpoint.pth', help="Path to a saved checkpoint to restart training.")

    # VAE Hyperparameters
    parser.add_argument("--latent_dim", type=int, default=8, help="Dimensionality of the latent space.")
    parser.add_argument("--encoder_dims", type=lambda s: [int(item) for item in s.split(',')], default="1024, 512, 256, 128", help="Comma-separated list of encoder dimensions.")
    parser.add_argument("--decoder_dims", type=lambda s: [int(item) for item in s.split(',')], default="128, 256, 512, 1024", help="Comma-separated list of decoder dimensions.")

    parser.add_argument("--lr_vae", type=float, default=1e-5, help="Learning rate for vae.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training.")

    # Training phase/epoch
    parser.add_argument("--warmup_epoch", type=int, default=200, help="Number of epochs for warmup.")
    parser.add_argument("--total_epochs", type=int, default=1000, help="Total number of training epochs.")
    parser.add_argument("--checkpoint_freq", type=int, default=5, help="Frequency of saving checkpoints.")
    parser.add_argument("--to_learning_freq", type=int, default=5, help="Frequency to switch phases.") 

    # TO Hyperparameters
    parser.add_argument("--zeta", type=float, default=0.01, help="Regularization parameter for sparsity.")
    parser.add_argument("--gamma", type=float, default=1e-3, help="Regularization parameter for dictionary.")
    parser.add_argument("--lr_eta_E", type=float, default=1e-5, help="Learning rate for TO E-step.")
    parser.add_argument("--lr_eta_M", type=float, default=1e-6, help="Learning rate for TO M-step.")
    parser.add_argument("--M", type=int, default=4, help="Some hyperparameter M.")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of iterations.")

    # Miscellaneous
    parser.add_argument("--WANDB_LOGGING", type=bool, default=True, help="Flag to enable or disable W&B logging.")
    parser.add_argument("--wandb_project_name", type=str, default="VAETO", help="W&B project name.")
    parser.add_argument("--output_dir", type=str, default='./out/debug', help="Path to save .pth files.")

    return parser

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = "cpu"
        
    print(f"Using device: {device}") 

    print(f"Start loading data")
    train_loader, input_dim = get_dataloader(data_path=args.data_path,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            header=None,  # Specify header as needed
                                            transform=None  # Specify transform function as needed
                                            )
    print("Data loaded successfully!")

    vae = VAE(
        input_dim=input_dim, 
        latent_dim=args.latent_dim,
        encoder_dims=args.encoder_dims, 
        decoder_dims=args.decoder_dims).to(device).float()
    print("VAE Architecture:")
    print(vae)

    transport_operator = TransportOperator(
        latent_dim = args.latent_dim,
        M = args.M,
        gamma = args.gamma,
        zeta = args.zeta,
        lr_eta_E = args.lr_eta_E,
        lr_eta_M = args.lr_eta_M).to(device).float() 
    print("Transport Operator initialization finished.")

    optimizer_vae = optim.Adam(vae.parameters(), lr=args.lr_vae)
    
    warmup_epoch = args.warmup_epoch
    chkpt_exists = os.path.exists(args.checkpoint_name)
    if not chkpt_exists:
        print("chkpt doesn't exist")
        start_epoch = 0
    else:
        print("chkpt exists")
        #checkpoint_path = os.path.join(args.output_dir, args.checkpoint_name)
        start_epoch = load_checkpoint(args.checkpoint_name, vae, device, optimizer_vae) + 1
        print(f"Resuming training from epoch {start_epoch}")

    vae.train()
    for epoch in range(start_epoch, args.total_epochs):

        # VAE PART    
        train_vaeto = epoch >= warmup_epoch
        train_loss, total_bce, total_kld, total_to_transformed_mse = train_vae(train_loader, 
                                                                                vae, 
                                                                                transport_operator, 
                                                                                optimizer_vae, 
                                                                                train_vaeto, 
                                                                                device, 
                                                                                args)
            
        # Logging to wandb
        if args.WANDB_LOGGING:
            wandb.log({"VAE_loss": train_loss / len(train_loader.dataset),
                       "VAE_BCE": total_bce / len(train_loader.dataset),
                       "VAE_KLD": total_kld / len(train_loader.dataset),
                       "VAE_mu_MSE": total_to_transformed_mse / len(train_loader.dataset)})
        if not train_vaeto:
            print(f'Epoch {epoch} (warmup, conventional vae), Loss: {train_loss / len(train_loader.dataset): .6f}')
        else:
            print(f'Epoch {epoch} (vae+to), Loss: {train_loss / len(train_loader.dataset): .6f}')

        if (epoch+1) == warmup_epoch:
            checkpoint = {'epoch': epoch,
                          'model_state_dict': vae.state_dict(),
                          'optimizer_state_dict': optimizer_vae.state_dict(),
                          'psi':transport_operator.psi}
            checkpoint_path = os.path.join(args.output_dir, f"vae_warmup_init.pth")
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, f"./checkpoint_init.pth")
        # elif not train_vaeto and ((epoch+1) % args.checkpoint_freq == 0):
        #     checkpoint = {'epoch': epoch,
        #                   'model_state_dict': vae.state_dict(),
        #                   'optimizer_state_dict': optimizer_vae.state_dict(),
        #                   'psi':transport_operator.psi}
        #     checkpoint_path = os.path.join(args.output_dir, f"vae_warmup_epoch{epoch}.pth")
        #     torch.save(checkpoint, checkpoint_path)

        # TO PART
        train_to = ((epoch+1) == warmup_epoch) or ((epoch+1) > warmup_epoch and (epoch+1 - warmup_epoch) % args.to_learning_freq == 0)
        if train_to:
            # Train transport operator
            total_energy, total_recon_loss, total_trans_op_loss, total_coef_loss = train_transport_operator(train_loader, 
                                                                                                            vae, 
                                                                                                            transport_operator, 
                                                                                                            train_vaeto,
                                                                                                            device, 
                                                                                                            args)
            # Logging to wandb
            if args.WANDB_LOGGING:
                wandb.log({"TO_energy": total_energy / len(train_loader.dataset),
                            "TO_recon_loss": total_recon_loss / len(train_loader.dataset),
                            "TO_trans_op": total_trans_op_loss / len(train_loader.dataset),
                            "TO_coef": total_coef_loss / len(train_loader.dataset)})
            print(f'Epoch {epoch} (trans_op learning), Loss: {total_energy / len(train_loader.dataset): .6f}')

        # Checkpoint saving
        if (epoch+1) == warmup_epoch:
            checkpoint = {'epoch': epoch,
                          'model_state_dict': vae.state_dict(),
                          'optimizer_state_dict': optimizer_vae.state_dict(),
                          'psi':transport_operator.psi}
            checkpoint_path = os.path.join(args.output_dir, f"vae_warmup_epoch{epoch}.pth")
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, f"./checkpoint.pth")
        elif train_vaeto and ((epoch+1-warmup_epoch) % args.checkpoint_freq == 0):
            checkpoint = {'epoch': epoch,
                          'model_state_dict': vae.state_dict(),
                          'optimizer_state_dict': optimizer_vae.state_dict(),
                          'psi':transport_operator.psi}
            checkpoint_path = os.path.join(args.output_dir, f"vae_epoch{epoch}.pth")
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, f"./checkpoint.pth")
 
if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    name = f"{current_time}"
    if args.WANDB_LOGGING:
        import wandb
        wandb.init(project=args.wandb_project_name, name = name)
        wandb.config.update(args)
    main(args)
