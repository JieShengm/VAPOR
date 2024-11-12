import torch
import torch.optim as optim

from pathlib import Path
import datetime
import os
from tqdm import tqdm

from .model import *
from .train import train_vae, train_transport_operator
from .utilities import get_dataloader, load_checkpoint

class VAPOR:
    def __init__(self,
                 data_path = './data/Synthetic/spiral5000.csv',
                 latent_dim = 3,
                 encoder_dims = [1024, 512, 256, 128],
                 decoder_dims = [128, 256, 512, 1024],
                 lr_vae = 1e-5,
                 batch_size = 512,
                 aggregation_type='direct_sum',
                 warmup_epoch = 100,
                 total_epochs = 500,
                 checkpoint_freq = 50,
                 to_learning_freq = 25,
                 zeta = 1e-4,
                 gamma = 1e-4,
                 lr_eta_E = 1e-5,
                 lr_eta_M = 1e-5,
                 M = 4,
                 max_iterations = 5000,
                 WANDB_LOGGING = False,
                 wandb_project_name = "VAPOR_DEBUG",
                 output_dir = './out/',
                 checkpoint_name = 'checkpoint.pth',
                 header = None,  
                 # transform = None  # Add other parameters as needed)
    ):
        self.config = dict(
            # Data Parameters
            data_path = data_path,
            # VAE Hyperparameters
            latent_dim = latent_dim,
            encoder_dims = encoder_dims,
            decoder_dims = decoder_dims,
            lr_vae = lr_vae,
            aggregation_type=aggregation_type,
            # Training phase/epoch
            batch_size = batch_size,
            warmup_epoch = warmup_epoch,
            total_epochs = total_epochs,
            checkpoint_freq = checkpoint_freq,
            to_learning_freq = to_learning_freq,
            # TO Hyperparameters
            zeta = zeta,
            gamma = gamma,
            lr_eta_E = lr_eta_E,
            lr_eta_M = lr_eta_M,
            M = M,
            max_iterations = max_iterations,
            # Miscellaneous
            WANDB_LOGGING = WANDB_LOGGING,
            wandb_project_name = wandb_project_name,
            output_dir = output_dir,
            checkpoint_name = checkpoint_name,
            header = header,  # Add other parameters as needed
            # 'transform' = transform  # Add other parameters as needed
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = None
        self.transport_operator = None
        self.optimizer_vae = None
        self.train_loader = None
        self.input_dim = None
        
        if self.config['WANDB_LOGGING']:
            import wandb
        
    def load_data(self, data_path=None):
        """Load data from specified path or use the path from initialization"""
        if data_path:
            self.config['data_path'] = data_path
            
        if not self.config['data_path']:
            raise ValueError("No data path specified")
            
        print(f"Start loading data")
        self.train_loader, self.input_dim = get_dataloader(
            data_path=self.config['data_path'],
            batch_size=self.config['batch_size'],
            shuffle=True,
            header=self.config['header'],
        )
        print("Data loaded successfully!")
        return self.train_loader
    
    def initialize_model(self):
        vae_classes = {
            'direct_sum': DirectSumVAE,
            'weighted_sum': WeightedSumVAE,
            'gated_sum': GatedSumVAE
        }
        VAEClass = vae_classes[self.config['aggregation_type']]
        
        """Initialize VAE and Transport Operator models"""
        if self.input_dim is None:
            raise ValueError("Please load data first using load_data()")
            
        self.vae = VAEClass(
            input_dim=self.input_dim,
            latent_dim=self.config['latent_dim'],
            psi_M=self.config['M'],
            encoder_dims=self.config['encoder_dims'],
            decoder_dims=self.config['decoder_dims']
        ).to(self.device).float()
        
        self.transport_operator = TransportOperator(
            latent_dim=self.config['latent_dim'],
            M=self.config['M'],
            gamma=self.config['gamma'],
            zeta=self.config['zeta'],
            lr_eta_E=self.config['lr_eta_E'],
            lr_eta_M=self.config['lr_eta_M']
        ).to(self.device).float()
        
        self.optimizer_vae = optim.Adam(self.vae.parameters(), lr=self.config['lr_vae'])
        
        return self.vae, self.transport_operator
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        if self.vae is None:
            raise ValueError("Please initialize model first using initialize_model()")
            
        start_epoch = load_checkpoint(checkpoint_path, self.vae, self.device, self.optimizer_vae) + 1
        print(f"Resumed from epoch {start_epoch}")
        return start_epoch
    
    def train(self, checkpoint_path=None):
        """Train the model with alternating VAE and TO phases"""
        if not self.vae or not self.transport_operator:
            if self.input_dim is None:
                raise ValueError("Please load data first using load_data()")
            print("Models not initialized. Initializing now...")
            self.initialize_model()
            
        if self.config['WANDB_LOGGING']:
            try:
                import wandb
                current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                wandb.init(project=self.config['wandb_project_name'], name=current_time)
                wandb.config.update(self.config)
            except ImportError:
                print("Warning: wandb not installed. Running without logging.")
                self.config['WANDB_LOGGING'] = False
                
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            start_epoch = self.load_checkpoint(checkpoint_path)
                
        self.vae.train()
        warmup_epoch = self.config['warmup_epoch']
        
        for epoch in range(start_epoch, self.config['total_epochs']):
            # VAE PART
            train_vaeto = epoch >= warmup_epoch
            
            print(f"\nEpoch {epoch} - Before VAE training:")
            print(f"Current TO psi shape: {self.transport_operator.psi.shape}")
            if hasattr(self.transport_operator, 'filtered_indices'):
                print(f"TO filtered_indices: {self.transport_operator.filtered_indices}")
        
            train_loss, total_bce, total_kld = train_vae(
                self.train_loader, 
                self.vae, 
                self.transport_operator, 
                self.optimizer_vae, 
                train_vaeto, 
                self.device, 
                self.config
            )
            
            # Log VAE results
            if self.config['WANDB_LOGGING']:
                wandb.log({
                    "VAE_loss": train_loss / len(self.train_loader.dataset),
                    "VAE_BCE": total_bce / len(self.train_loader.dataset),
                    "VAE_KLD": total_kld / len(self.train_loader.dataset)
                })
                
            print(f'Epoch {epoch} ' + 
                f'({("vae+to" if train_vaeto else "warmup, conventional vae")}), ' +
                f'Loss: {train_loss / len(self.train_loader.dataset):.6f}')

            # Save warmup checkpoint
            if (epoch + 1) == warmup_epoch:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.vae.state_dict(),
                    'optimizer_state_dict': self.optimizer_vae.state_dict(),
                    'psi': self.transport_operator.psi
                }
                checkpoint_path = os.path.join(self.config['output_dir'], "vae_warmup_init.pth")
                torch.save(checkpoint, checkpoint_path)

            # TO PART
            train_to = ((epoch + 1) == warmup_epoch) or (
                (epoch + 1) > warmup_epoch and 
                (epoch + 1 - warmup_epoch) % self.config['to_learning_freq'] == 0
            )
            
            if train_to:
                # Train transport operator
                total_energy, total_recon_loss, total_trans_op_loss, total_coef_loss = train_transport_operator(
                    self.train_loader,
                    self.vae,
                    self.transport_operator,
                    train_vaeto,
                    self.device,
                    self.config
                )
                
                # Log TO results
                if self.config['WANDB_LOGGING']:
                    wandb.log({
                        "TO_energy": total_energy / len(self.train_loader.dataset),
                        "TO_recon_loss": total_recon_loss / len(self.train_loader.dataset),
                        "TO_trans_op": total_trans_op_loss / len(self.train_loader.dataset),
                        "TO_coef": total_coef_loss / len(self.train_loader.dataset)
                    })
                print(f'Epoch {epoch} (trans_op learning), ' + 
                    f'Loss: {total_energy / len(self.train_loader.dataset):.6f}')

            # Save checkpoints
            if (epoch + 1) == warmup_epoch:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.vae.state_dict(),
                    'optimizer_state_dict': self.optimizer_vae.state_dict(),
                    'psi': self.transport_operator.psi
                }
                checkpoint_path = os.path.join(
                    self.config['output_dir'], 
                    f"vae_warmup_epoch{epoch}.pth"
                )
                torch.save(checkpoint, checkpoint_path)
            elif train_vaeto and ((epoch + 1 - warmup_epoch) % self.config['checkpoint_freq'] == 0):
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.vae.state_dict(),
                    'optimizer_state_dict': self.optimizer_vae.state_dict(),
                    'psi': self.transport_operator.psi
                }
                checkpoint_path = os.path.join(
                    self.config['output_dir'], 
                    f"vae_epoch{epoch}.pth"
                )
                torch.save(checkpoint, checkpoint_path)

    def _handle_logging_and_checkpoints(self, epoch, train_loss, total_bce, total_kld,
                                      train_vaeto, train_loader):
        """Handle logging and checkpoint saving"""
        dataset_size = len(train_loader.dataset)
        
        if self.config['WANDB_LOGGING']:
            import wandb
            wandb.log({
                "VAE_loss": train_loss / dataset_size,
                "VAE_BCE": total_bce / dataset_size,
                "VAE_KLD": total_kld / dataset_size
            })
            
        print(f'Epoch {epoch} ' + 
              f'({"vae+to" if train_vaeto else "warmup, conventional vae"}), ' +
              f'Loss: {train_loss / dataset_size:.6f}')
        
        # Save checkpoints
        if self._should_save_checkpoint(epoch):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.vae.state_dict(),
                'optimizer_state_dict': self.optimizer_vae.state_dict(),
                'psi': self.transport_operator.psi
            }
            
            checkpoint_path = os.path.join(
                self.config['output_dir'],
                f"vae_{'warmup_' if epoch + 1 == self.config['warmup_epoch'] else ''}epoch{epoch}.pth"
            )
            torch.save(checkpoint, checkpoint_path)
            
    def _should_save_checkpoint(self, epoch):
        """Determine if checkpoint should be saved"""
        is_warmup_end = (epoch + 1) == self.config['warmup_epoch']
        is_checkpoint_freq = ((epoch + 1 - self.config['warmup_epoch']) % 
                            self.config['checkpoint_freq'] == 0)
        return is_warmup_end or (epoch >= self.config['warmup_epoch'] and is_checkpoint_freq)

    def hyperparameter_search(self, 
                            param_grid=None,
                            n_trials=50,
                            metric='loss',
                            search_method='random',  # 'random' or 'grid'
                            n_epochs_per_trial=10):  # Quick evaluation for each trial
        """
        Perform hyperparameter search and generate analysis report
        
        Parameters:
        -----------
        param_grid : dict
            Dictionary of parameters to search. If None, use default search space.
            Example: {
                'latent_dim': [2, 3, 5, 8],
                'lr_vae': [1e-4, 1e-3, 1e-2],
                'batch_size': [128, 256, 512],
                'encoder_dims': [[512, 256], [1024, 512, 256]],
                'M': [2, 4, 8],
                'zeta': [0.001, 0.01, 0.1],
                'gamma': [1e-4, 1e-3, 1e-2]
            }
        """
        import optuna
        from collections import defaultdict
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Default search space if none provided
        if param_grid is None:
            param_grid = {
                'latent_dim': [2, 3, 5, 8],
                'lr_vae': [1e-4, 1e-3, 1e-2],
                'batch_size': [128, 256, 512],
                'M': [2, 4, 8],
                'zeta': [0.001, 0.01, 0.1],
                'gamma': [1e-4, 1e-3, 1e-2]
            }
            
        results = defaultdict(list)
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_values in param_grid.items():
                if isinstance(param_values[0], (int, float)):
                    if all(isinstance(x, int) for x in param_values):
                        params[param_name] = trial.suggest_int(
                            param_name, min(param_values), max(param_values)
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, min(param_values), max(param_values), log=True
                        )
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            # Update model with new parameters
            self.config.update(params)
            
            # Quick evaluation
            try:
                self.initialize_model()  # Reinitialize with new params
                total_loss = 0
                
                # Train for few epochs to get quick estimate
                for epoch in range(n_epochs_per_trial):
                    loss = self._quick_evaluate()
                    total_loss += loss
                
                avg_loss = total_loss / n_epochs_per_trial
                
                # Store results
                for param_name, param_value in params.items():
                    results[param_name].append(param_value)
                results['loss'].append(avg_loss)
                
                return avg_loss
                
            except Exception as e:
                print(f"Trial failed with parameters {params}: {str(e)}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Generate report
        self._generate_search_report(results_df, study)
        
        return study.best_params, results_df
    
    def _quick_evaluate(self):
        """Quick evaluation of current parameters"""
        self.vae.train()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in self.train_loader:
                if isinstance(batch, (tuple, list)):
                    data = batch[0]
                else:
                    data = batch
                    
                data = data.to(self.device).float()
                
                # Forward pass
                recon_data, mu, log_var = self.vae(data)
                
                # Calculate loss
                recon_loss = torch.nn.functional.mse_loss(recon_data, data)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kl_loss
                
                total_loss += loss.item()
                n_batches += 1
                
                if n_batches >= 10:  # Limit evaluation to 10 batches for speed
                    break
        
        return total_loss / n_batches
    
    def _generate_search_report(self, results_df, study):
        """Generate comprehensive hyperparameter search report"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create output directory for report
        report_dir = os.path.join(self.config['output_dir'], 'hyperparam_search')
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. Summary statistics
        summary = results_df.describe()
        summary.to_csv(os.path.join(report_dir, 'summary_statistics.csv'))
        
        # 2. Parameter importances
        importances = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(10, 6))
        plt.bar(importances.keys(), importances.values())
        plt.xticks(rotation=45)
        plt.title('Parameter Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'parameter_importances.png'))
        plt.close()
        
        # 3. Correlation analysis
        plt.figure(figsize=(12, 10))
        sns.heatmap(results_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Parameter Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'correlations.png'))
        plt.close()
        
        # 4. Parameter vs Loss plots
        for param in results_df.columns:
            if param != 'loss':
                plt.figure(figsize=(8, 6))
                plt.scatter(results_df[param], results_df['loss'])
                plt.xlabel(param)
                plt.ylabel('Loss')
                plt.title(f'{param} vs Loss')
                plt.tight_layout()
                plt.savefig(os.path.join(report_dir, f'{param}_vs_loss.png'))
                plt.close()
        
        # 5. Generate HTML report
        report_html = f"""
        <html>
        <head>
            <title>Hyperparameter Search Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 30px; }}
                img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Hyperparameter Search Report</h1>
            
            <div class="section">
                <h2>Best Parameters</h2>
                <pre>{study.best_params}</pre>
                <p>Best loss: {study.best_value:.4f}</p>
            </div>
            
            <div class="section">
                <h2>Parameter Importances</h2>
                <img src="parameter_importances.png">
            </div>
            
            <div class="section">
                <h2>Parameter Correlations</h2>
                <img src="correlations.png">
            </div>
            
            <div class="section">
                <h2>Parameter vs Loss Relationships</h2>
                {' '.join(f'<img src="{param}_vs_loss.png">' 
                         for param in results_df.columns if param != 'loss')}
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <pre>{summary.to_html()}</pre>
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(report_dir, 'report.html'), 'w') as f:
            f.write(report_html)
        
        print(f"\nHyperparameter search report generated in {report_dir}")
        print(f"Best parameters found: {study.best_params}")
        print(f"Best loss achieved: {study.best_value:.4f}")