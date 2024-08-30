from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import matplotlib as plt
from scipy.sparse import issparse

class Dataset(Dataset):
    def __init__(self, data_path, header=None, transform=None):
        self.file = None
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path, header=header).to_numpy()
            print(f'n_OBS: {self.data.shape[0]}; n_VAR: {self.data.shape[1]}')
        elif data_path.endswith('.h5ad'):
            import anndata as ad
            h5ad_data = ad.read_h5ad(data_path)
            if issparse(h5ad_data.X):
                self.data = h5ad_data.X.toarray()
            print(f'n_OBS: {self.data.shape[0]}; n_VAR: {self.data.shape[1]}')
        else:
            raise AssertionError("The file format is not supported/available now.")

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # if self.file == 'csv':
        #     sample = self.data[idx, :]
        # elif self.file == 'h5ad':
        #     sample = self.data[idx].toarray() #slow
        sample = self.data[idx, :]
        
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_dataloader(data_path, batch_size, shuffle=True,  header=None, transform=None, return_input_dim = True):
    dataset = Dataset(data_path=data_path, header=header, transform=transform)
    input_dim = dataset.data.shape[1]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last = True)
    if return_input_dim:
        return dataloader, input_dim
    else:
        return dataloader 

def load_checkpoint(checkpoint_path, model, device, optimizer=None):
    """
    Loads model state (and optimizer state) from a file.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', -1)  # Return the last completed epoch number, if available

# def plot_pairs(z0, z1):
#     fig = plt.figure(figsize=(25, 7.5))
#     ax1 = fig.add_subplot(131)
#     ax1.scatter(z0[:,0].cpu(), z0[:,1].cpu())
#     ax1.set_title('X0')

#     ax2 = fig.add_subplot(132)
#     ax2.scatter(z1[:,0].cpu(), z1[:,1].cpu())
#     ax2.set_title('X1')
    
#     ax3 = fig.add_subplot(133)
#     ax3.scatter(z0[:,0].cpu(), z0[:,1].cpu(), color='blue', s=10)  
#     ax3.scatter(z1[:,0].cpu(), z1[:,1].cpu(), color='red', s=10)   
#     for i in range(len(z0)):
#         ax3.plot([z0[i,0].cpu(), z1[i,0].cpu()], [z0[i,1].cpu(), z1[i,1].cpu()], color='green')  
#     ax3.set_title('Correspondence')
#     plt.savefig("paired_points_visualization.png")  
#     plt.close(fig) 
#     return 