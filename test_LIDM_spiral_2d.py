import pickle

import numpy as np
import torch
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader
from utils import *
import LIDM
from LIDM import *
spiral2d_dataset = pickle.load(open("./spiral2d-dataset/spiral2d_dataset.p", "rb"))

''' data'''
x = np.concatenate([spiral2d_dataset['observation'][:,0,:10].squeeze(),
                    spiral2d_dataset['observation'][:,1,:10].squeeze()], axis=-1)
z = spiral2d_dataset['state']


''' normalization'''
x = 2*(x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))-1
z = 2*(z-z.min(axis=0))/(z.max(axis=0)-z.min(axis=0))-1

device = torch.device('cuda')
Dataset = get_dataset(x, z, device)
Dataset_loader = DataLoader(Dataset, batch_size=x.shape[0],shuffle=False)
model = LIDM(latent_dim=z.shape[1], obser_dim=x.shape[1], sigma_x=.3,alpha=.1, importance_sample_size=1, n_layers=2,
              device=device).to(device)
model.apply(init_weights)
print(f'The g_theta model has {count_parameters(model.g_theta):,} trainable parameters')
print(f'The f_phi model has {count_parameters(model.f_phi):,} trainable parameters')
print(f'The f_phi.f_phi_x model has {count_parameters(model.f_phi.f_phi_x):,} trainable parameters')
print(f'The LIDM model has {count_parameters(model):,} trainable parameters')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
CLIP = 1
total_loss=[]
Numb_Epochs=500
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
z = z.detach().cpu().numpy().squeeze()
x_n = x.detach().cpu().numpy().squeeze()
save_result_path='Results/'
plt.figure()
plt.scatter(x_n[:, :10], x_n[:, 10:], marker='o', c='k', alpha=.1)
plt.plot(z[:, 0],z[:, 1],
         'k', label='true trajectory')
plt.scatter(z[:, 0], z[:, 1], label='noisy state', marker='o', c='k', s=20)
plt.savefig(save_result_path + 'spiral-data.png')
plt.savefig(save_result_path + 'spiral-data.svg', format='svg')
plt.show()
plt.figure()
plt.plot(total_loss)
plt.show()




plt.figure()
trj_samples=np.random.randint(0, 1,1)
for ii in trj_samples:
    z_hat = model(x, True)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:,:]
    z_hat =2*(z_hat-z_hat.min(axis=0))/(z_hat.max(axis=0)-z_hat.min(axis=0))-1
    plt.scatter(z_hat[:,0],z_hat[:,1], c='r')
    plt.plot(z_hat[:, 0], z_hat[:, 1], 'r')
plt.title('with observation')
x_hat=model.x_hat.cpu().detach().numpy().squeeze()
plt.scatter(x_hat[:,:10],x_hat[:,10:],c='k', alpha=.1)
z_x_hat=model.z_x_hat.cpu().detach().numpy().squeeze()
z_x_hat=2*(z_x_hat-z_x_hat.min(axis=0))/(z_x_hat.max(axis=0)-z_x_hat.min(axis=0))-1
plt.scatter(z_x_hat[:,0],z_x_hat[:,1:],c='g', alpha=.3)
plt.plot(z_x_hat[:,0],z_x_hat[:,1],c='g', alpha=1)
plt.plot(z[:,0], z[:,1], 'k')
plt.scatter(z[:,0],z[:,1:],c='k', alpha=1)
plt.savefig(save_result_path + 'spiral-with-obsr.png')
plt.savefig(save_result_path + 'spiral-with-obsr.svg', format='svg')
plt.show()

mean_true=get_cdf(x_n.mean(axis=1))
mean_pred=get_cdf(x_hat.mean(axis=1))
plt.figure()
plt.plot(mean_true,mean_pred,'r')
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('with observations')
plt.show()
plt.savefig(save_result_path + 'cdf-spiral-with-obsr.png')
plt.savefig(save_result_path + 'cdf-spiral-with-obsr.svg', format='svg')
print('LDIDPs-with-obsr-cc=%f,mse=%f,mae=%f'%(get_metrics(z[1:], z_hat)))
print('F-theta-with-obsr-cc=%f,mse=%f,mae=%f'%(get_metrics(z, z_x_hat)))

plt.figure()
for ii in trj_samples:
    z_hat = model(x, False)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:,:]
    z_hat = 2*(z_hat-z_hat.min(axis=0))/(z_hat.max(axis=0)-z_hat.min(axis=0))-1
    plt.scatter(z_hat[:,0],z_hat[:,1],c='b')
    plt.plot(z_hat[:, 0], z_hat[:, 1], 'b')
plt.title('No observation')

x_hat=model.x_hat.cpu().detach().numpy().squeeze()
plt.scatter(x_hat[:,:10],x_hat[:,10:],c='k', alpha=.1)
z_x_hat=model.z_x_hat.cpu().detach().numpy().squeeze()
z_x_hat=2*(z_x_hat-z_x_hat.min(axis=0))/(z_x_hat.max(axis=0)-z_x_hat.min(axis=0))-1
plt.scatter(z_x_hat[:,0],z_x_hat[:,1:],c='g', alpha=.3)
plt.plot(z_x_hat[:,0],z_x_hat[:,1],c='g', alpha=1)
plt.plot(z[:,0], z[:,1], 'k')
plt.scatter(z[:,0],z[:,1:],c='k', alpha=1)
plt.savefig(save_result_path + 'spiral-with-no-obsr.png')
plt.savefig(save_result_path + 'spiral-with-no-obsr.svg', format='svg')
plt.show()
mean_true=get_cdf(x_n.mean(axis=1))
mean_pred=get_cdf(x_hat.mean(axis=1))
plt.figure()
plt.plot(mean_true,mean_pred, 'b')
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('with-no-observations')
plt.show()
plt.savefig(save_result_path + 'cdf-spiral-with-no-obsr.png')
plt.savefig(save_result_path + 'cdf-spiral-with-no-obsr.svg', format='svg')
print('LDIDPs-no-obsr-cc=%f,mse=%f,mae=%f'%(get_metrics(z[1:], z_hat)))
print('F-theta-no-obsr-cc=%f,mse=%f,mae=%f'%(get_metrics(z, z_x_hat)))


''' performance measures'''


