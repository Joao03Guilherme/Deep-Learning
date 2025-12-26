import torch
from torch.utils.data import DataLoader
from utils import load_rnacompete_data

# 1. Load Data for a specific protein (e.g., 'RBFOX1', 'PTB', 'A1CF')
# This returns a PyTorch TensorDataset ready for training
train_dataset = load_rnacompete_data(protein_name='RBFOX1', split='train')
val_dataset   = load_rnacompete_data(protein_name='RBFOX1', split='val')
test_dataset  = load_rnacompete_data(protein_name='RBFOX1', split='test')

# 2. Wrap in a standard PyTorch DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

