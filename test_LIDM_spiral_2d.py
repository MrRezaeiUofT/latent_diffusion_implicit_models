import pickle

import numpy as np
import torch
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader

import LIDM
from LIDM import *
spiral2d_dataset = pickle.load(open("./spiral2d-dataset/spiral2d_dataset.p", "rb"))

''' data'''
x = np.concatenate([spiral2d_dataset['observation'][:,0,:10].squeeze(),
                    spiral2d_dataset['observation'][:,1,:10].squeeze()], axis=-1)
z = spiral2d_dataset['state']


''' normalization'''
x = 2*(x-x.mean(axis=0))/(x.max(axis=0)-x.min(axis=0))-1
z = 2*(z-z.min(axis=0))/(z.max(axis=0)-z.min(axis=0))-1

device = torch.device('cuda')
Dataset = get_dataset(x, z, device)
Dataset_loader = DataLoader(Dataset, batch_size=x.shape[0],shuffle=False)
model = LIDM(latent_dim=z.shape[1], obser_dim=x.shape[1], sigma_x=2,alpha=.5,
             time_lenth=x.shape[0], device=device).to(device)
model.apply(init_weights)
print(f'The g_theta model has {count_parameters(model.g_theta):,} trainable parameters')
print(f'The f_phi model has {count_parameters(model.f_phi):,} trainable parameters')
print(f'The f_phi.f_phi_x model has {count_parameters(model.f_phi.f_phi_x):,} trainable parameters')
print(f'The LIDM model has {count_parameters(model):,} trainable parameters')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
CLIP = 1
total_loss=[]
Numb_Epochs=100
for epoch in range(Numb_Epochs):
    epoch_loss = 0
    for i, batch in enumerate(Dataset_loader):
        x, z = batch
        x = torch.unsqueeze(x, 1)
        z = torch.unsqueeze(z, 1)

        optimizer.zero_grad()
        z_hat = model(x,True)
        print('epoch=%d/%d'%(epoch,Numb_Epochs))
        loss=model.loss(a=1,b=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        # optimizer.zero_grad()
        # z_hat = model(x, True)
        # loss = model.loss(a=0, b=1)
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        # optimizer.step()

        epoch_loss += loss.item()
    total_loss.append(epoch_loss)



import matplotlib.pyplot as plt

''' visualization'''
plt.figure()
plt.scatter(spiral2d_dataset['observation'][:, 0, :], spiral2d_dataset['observation'][:, 1, :], marker='o', c='k', s=3, alpha=.1)
plt.plot(spiral2d_dataset['state'][:, 0], spiral2d_dataset['state'][:, 1],
         'g', label='true trajectory')
plt.scatter(spiral2d_dataset['state'][:, 0], spiral2d_dataset['state'][:, 1], label='noisy state', marker='*', c='r', s=20)
plt.show()
plt.figure()
plt.plot(total_loss)
plt.show()

z = z.detach().cpu().numpy().squeeze()
plt.figure()


trj_samples=np.random.randint(2, 10,1)
for ii in trj_samples:
    z_hat = model(x, True)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:, trj_samples, :].squeeze()
    # z_hat =z_hat#2*(z_hat-z_hat.min(axis=0))/(z_hat.max(axis=0)-z_hat.min(axis=0))-1
    plt.scatter(z_hat[:,0],z_hat[:,1], c='r', alpha=.5)
    plt.plot(z_hat[:, 0], z_hat[:, 1], 'r')
plt.title('with observation')
# plt.plot(z[:,0], z[:,1], 'k')
plt.show()

plt.figure()
for ii in trj_samples:
    z_hat = model(x, False)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:, trj_samples, :].squeeze()
    # z_hat = z_hat#2*(z_hat-z_hat.min(axis=0))/(z_hat.max(axis=0)-z_hat.min(axis=0))-1
    plt.scatter(z_hat[:,0],z_hat[:,1],c='b', alpha=.5)
    plt.plot(z_hat[:, 0], z_hat[:, 1], 'b')
    plt.plot(z_hat[:, 0], z_hat[:, 1], 'b')
plt.title('No observation')
# plt.plot(z[:,0], z[:,1], 'k')
plt.show()