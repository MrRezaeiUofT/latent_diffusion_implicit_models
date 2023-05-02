import pickle
import torch
import torch.optim as optim

from torch.utils.data import DataLoader

import LIDM
from LIDM import *
spiral2d_dataset = pickle.load(open("./spiral2d-dataset/spiral2d_dataset.p", "rb"))

x = spiral2d_dataset['observation'].reshape([spiral2d_dataset['observation'].shape[0],-1]).squeeze()
z = spiral2d_dataset['state']
''' normalization'''
x = 2*(x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))-1
z = 2*(z-z.min(axis=0))/(z.max(axis=0)-z.min(axis=0))-1

device = torch.device('cuda')
Dataset = get_dataset(x, z, device)
Dataset_loader = DataLoader(Dataset, batch_size=x.shape[0],shuffle=False)
model = LIDM(latent_dim=z.shape[1], obser_dim=x.shape[1], sigma_x=.1,sigma_z=.1,alpha=1, device=device).to(device)
model.apply(init_weights)
print(f'The g_theta model has {count_parameters(model.g_theta):,} trainable parameters')
print(f'The f_phi model has {count_parameters(model.f_phi):,} trainable parameters')
print(f'The f_phi.f_phi_x model has {count_parameters(model.f_phi.f_phi_x):,} trainable parameters')
print(f'The LIDM model has {count_parameters(model):,} trainable parameters')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
CLIP = 1
total_loss=[]
for epoch in range(250):
    epoch_loss = 0
    for i, batch in enumerate(Dataset_loader):
        x, z = batch
        x = torch.unsqueeze(x, 1)
        z = torch.unsqueeze(z, 1)

        optimizer.zero_grad()

        z_hat = model(x)
        loss=model.loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        optimizer.step()

        epoch_loss += loss.item()
    total_loss.append(epoch_loss)

import matplotlib.pyplot as plt
plt.plot(total_loss)
plt.show()
plt.figure()
z_hat=z_hat.detach().cpu().numpy().squeeze()
z=z.detach().cpu().numpy().squeeze()
z_hat = 2*(z_hat-z_hat.min(axis=0))/(z_hat.max(axis=0)-z_hat.min(axis=0))-1
plt.plot(z_hat[:,0],z_hat[:,1], 'r')
plt.plot(z[:,0], z[:,1], 'k')
plt.show()