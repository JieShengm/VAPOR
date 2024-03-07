import argparse
import os
import datetime
from pathlib import Path

import torch
import torch.optim as optim

from model import VAE, TransportOperator, vae_loss
from utilities import get_dataloader, load_checkpoint


def get_args_parser():
    parser = argparse.ArgumentParser(description="Train VAE.")
    
    parser.add_argument("--data_path", type=str, default='./data/Pasca/train_pasca_subset_expr.csv', help="Data Path.")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="Path to a saved checkpoint to restart training.")

    # VAE Hyperparameters
    parser.add_argument("--latent_dim", type=int, default=8, help="Dimensionality of the latent space.")
    parser.add_argument("--encoder_dims", type=lambda s: [int(item) for item in s.split(',')], default="1024,512,256,128", help="Comma-separated list of encoder dimensions.")
    parser.add_argument("--decoder_dims", type=lambda s: [int(item) for item in s.split(',')], default="128,256,512,1024", help="Comma-separated list of decoder dimensions.")

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--warmup_epoch", type=int, default=20, help="Number of epochs for warmup.")
    parser.add_argument("--total_epochs", type=int, default=200, help="Total number of training epochs.")
    parser.add_argument("--checkpoint_freq", type=int, default=200, help="Frequency of saving checkpoints.")
    parser.add_argument("--phase_switch_freq", type=int, default=20, help="Frequency to switch phases.")
    
    # TO Hyperparameters
    parser.add_argument("--zeta", type=float, default=0.01, help="Regularization parameter for sparsity.")
    parser.add_argument("--gamma", type=float, default=1e-8, help="Regularization parameter for dictionary.")
    parser.add_argument("--eta", type=float, default=0.0001, help="Learning rate for TO.")
    parser.add_argument("--M", type=int, default=3, help="Some hyperparameter M.")
    parser.add_argument("--max_iterations", type=int, default=5, help="Maximum number of iterations.")

    # Miscellaneous
    parser.add_argument("--WANDB_LOGGING", type=bool, default=False, help="Flag to enable or disable W&B logging.")
    parser.add_argument("--wandb_project_name", type=str, default="VAETO", help="W&B project name.")
    parser.add_argument("--output_dir", type=str, default='./out_test', help="Path to save .pth files.")

    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        M = args.M).to(device).float() 
    print("Transport Operator initialization finished.")

    optimizer_vae = optim.Adam(vae.parameters(), lr=args.lr)
    #optimizer_to = torch.optim.Adam(transport_operator.parameters(), lr=...)

    start_epoch = 0
    if args.checkpoint_name:
        checkpoint_path = os.path.join(args.output_dir, args.checkpoint_name)
        start_epoch = load_checkpoint(checkpoint_path, vae, device, optimizer_vae) + 1
        print(f"Resuming training from epoch {start_epoch}")

    vae.train()
    for epoch in range(start_epoch, args.total_epochs):
        if epoch < args.warmup_epoch:
            train_vae = True
        else:
            if epoch % args.phase_switch_freq == 0:
                train_vae = not train_vae

        if train_vae:
            print(f'Train VAE: {epoch} - {train_vae}')
            # # Train VAE
            # train_loss, total_bce, total_kld = 0, 0, 0
            # for _, data in enumerate(train_loader):
            #     data = data.float().to(device)
            #     optimizer_vae.zero_grad()

            #     if epoch < args.warmup_epoch:
            #         # Warmup phase: Train as a conventional VAE
            #         recon, mu, logvar = vae(data)
            #         BCE, KLD = vae_loss(data, recon, mu, logvar)
            #         loss = BCE + KLD
            #     loss.backward()
            
            #     train_loss += loss.item()
            #     total_bce += BCE.item()
            #     total_kld += KLD.item()
            #     optimizer_vae.step()

            # # Logging to wandb
            # if args.WANDB_LOGGING:
            #     wandb.log({"loss": train_loss / len(train_loader.dataset),
            #                 "BCE": total_bce / len(train_loader.dataset),
            #                 "KLD": total_kld / len(train_loader.dataset)})

            # # Checkpoint saving
            # if (epoch+1) % args.checkpoint_freq == 0:
            #     checkpoint = {
            #         'epoch': epoch,
            #         'model_state_dict': vae.state_dict(),
            #         'optimizer_state_dict': optimizer_vae.state_dict(),
            #         }
            #     checkpoint_path = os.path.join(args.output_dir, f"vae_{epoch}.pth")
            #     torch.save(checkpoint, checkpoint_path)

            # print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset): .6f}')

        else:
            # Train transport operator
            print(f'Train TO: {epoch} - {train_vae}')
            #total_energy, total_recon, total_fro, total_c = 0, 0, 0, 0
            for _, data in enumerate(train_loader):
                data = data.float().to(device)
                with torch.no_grad():
                    z0 = vae.forward(data, return_latent = True)
                    #print(z0) # check whether no_grad(): yes
                pairs = transport_operator.construct_pairs(z0)
                print(pairs[0])
                print(pairs[0].shape)
                # TO_leanring(), optimizing steps
                
                # to get psi, and coef

                # get new z0_ast{?}, recond_batch_ast?

                #print(pairs)

                break
                # Train the TO with z0, mu, logvar, or a

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    entity_name = f"{current_time}"
    if args.WANDB_LOGGING:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=entity_name)
        wandb.config.update(args)

    main(args)
