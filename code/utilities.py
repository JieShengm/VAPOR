from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

class Dataset(Dataset):
    def __init__(self, data_path, header=None, transform=None):
        self.file = None
        # self.data = pd.read_csv(data_path, header=header).to_numpy()
        if data_path.endswith('.csv'):
            # self.file = 'csv'
            self.data = pd.read_csv(data_path, header=header).to_numpy()
            print(f'n_OBS: {self.data.shape[0]}; n_VAR: {self.data.shape[1]}')
        elif data_path.endswith('.h5ad'):
            import anndata as ad
            # self.file = 'h5ad'
            h5ad_data = ad.read_h5ad(data_path)
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
