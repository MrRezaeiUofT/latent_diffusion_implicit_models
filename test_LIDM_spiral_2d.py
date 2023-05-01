import pickle
import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from LIDM import *
spiral2d_dataset = pickle.load(open("/Users/mohammad.rezaei/PycharmProjects/latent_diffusion_implicit_models/spiral2d-dataset/spiral2d_dataset.p", "rb"))

x = spiral2d_dataset['observation'].reshape([spiral2d_dataset['observation'].shape[0],-1])
z = spiral2d_dataset['state']
device = torch.device('cpu')
model = LIDM(latent_dim=2, obser_dim=6, device=device).to(device)
model.apply(init_weights)
# print(f'The LIDM model has {count_parameters(LIDM):,} trainable parameters')
# optimizer = optim.Adam(model.parameters(), lr=1e-2)
