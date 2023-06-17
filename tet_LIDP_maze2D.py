from scipy import io
from scipy import stats
import sys
import numpy as np
import pickle
import torch
import torch.optim as optim
from utils import *
# torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader
import scipy.io as sio

from utils import *
import LDIDP_maze
from LDIDP_maze import *
import torch
from torch.distributions import MultivariateNormal

folder='./NeuralData/' #ENTER THE FOLDER THAT YOUR DATA IS IN
save_result_path='Results/'

Data=sio.loadmat('./NeuralData/Data33.mat')['MSTrain']
# Data=Data[-2000:,:]
NeuralData=Data[:,1:63]
bestIndxs=calInformetiveChan(NeuralData,100)


YCells=calSmoothNeuralActivity(np.squeeze(NeuralData[:,bestIndxs]),100,8)

# normalization
for iii in range(YCells.shape[-1]):
    YCells[:,iii]=(YCells[:,iii]-YCells[:,iii].mean())/YCells[:,iii].std()


Xs=2*((Data[:,65]-np.min(Data[:,65]))/(np.max(Data[:,65])-np.min(Data[:,65]))-.5)
Ys=2*((Data[:,66]-np.min(Data[:,66]))/(np.max(Data[:,66])-np.min(Data[:,66]))-.5)
Y=np.concatenate([Xs.reshape([-1,1]),Ys.reshape([-1,1])],axis=-1)





downsample_rate=10
x_all=YCells[np.arange(0, YCells.shape[0], downsample_rate),:]
x_all = calDesignMatrix_V2(x_all,3).squeeze()
z_all=Y[np.arange(0, YCells.shape[0], downsample_rate),:]
x_all = 2*(x_all-x_all.min(axis=0))/(x_all.max(axis=0)-x_all.min(axis=0))-1
# import matplotlib.pyplot as plt
# f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
# axes[0].plot(z[:, 0].squeeze(), 'k')
# axes[1].plot(z[:, 1].squeeze(), 'k')
# plt.show()
# [200,400,600,750,900,1050,1180,1300,1430,1550,2100,2250,2430,2480]
str_ind=[200,400,600,750,900,1050,1180,1300,1430,1550,2100,2250,2430,2480]
trial_length=250
x=np.zeros((trial_length,len(str_ind), x_all.shape[1]))
z=np.zeros((trial_length,len(str_ind), z_all.shape[1]))
for ii in range(len(str_ind)):
    x[:,ii,:]=x_all[str_ind[ii]:str_ind[ii]+trial_length,:]
    z[:, ii, :] = z_all[str_ind[ii]:str_ind[ii] + trial_length, :]

# import matplotlib.pyplot as plt
# f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
# axes[0].plot(z[:,0, 0].squeeze(), 'k')
# axes[1].plot(z[:,0, 1].squeeze(), 'k')
# plt.show()




device = torch.device('cuda')
Dataset = get_dataset(x, z, device)
Dataset_loader = DataLoader(Dataset, batch_size=x.shape[0], shuffle=False)
model = LIDM(latent_dim=z.shape[-1], obser_dim=x.shape[-1], sigma_x=.1, alpha=.2,importance_sample_size=1, n_layers=4,
              device=device).to(device)
model.apply(init_weights)
print(f'The g_theta model has {count_parameters(model.g_theta):,} trainable parameters')
print(f'The f_phi model has {count_parameters(model.f_phi):,} trainable parameters')
print(f'The f_phi.f_phi_x model has {count_parameters(model.f_phi.f_phi_x):,} trainable parameters')
print(f'The LIDM model has {count_parameters(model):,} trainable parameters')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
CLIP = 1
total_loss = []
Numb_Epochs = 1
for epoch in range(Numb_Epochs):
    epoch_loss = 0
    for i, batch in enumerate(Dataset_loader):
        x, z = batch
        # print(x.shape)
        # x = torch.swapaxes(x, 0,1)
        # z = torch.swapaxes(z, 0,1)
        optimizer.zero_grad()
        z_hat = model(x,z, True)
        print('epoch=%d/%d' % (epoch, Numb_Epochs))
        loss = model.loss(a=1, b=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        epoch_loss += loss.item()
    total_loss.append(epoch_loss)

import matplotlib.pyplot as plt

''' visualization'''
# plt.figure()
# plt.imshow(x.cpu().detach().numpy().squeeze().T)
# plt.show()

plt.figure()
plt.plot(total_loss)
plt.show()
plt.savefig(save_result_path + 'loss-maze-with-no-obsr.png')
plt.savefig(save_result_path + 'loss-maze-with-no-obsr.svg', format='svg')
x_n = x.detach().cpu().numpy().squeeze()
z_tr = z.detach().cpu().numpy().squeeze()
# plt.figure()

# ax = plt.axes(projection='3d')
trj_samples = np.arange(0,x.shape[1])
for ii in trj_samples:
    z_hat = model(x, z, True)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:,ii, :]

    z_hat = 2 * (z_hat - z_hat.min(axis=0)) / (z_hat.max(axis=0) - z_hat.min(axis=0)) - 1
    f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
    axes[0].plot(z_hat[:, 0].squeeze(), 'r')
    axes[0].plot(z_tr[1:,ii, 0].squeeze(), 'k')

    axes[1].plot(z_hat[:, 1].squeeze(), 'r')
    axes[1].plot(z_tr[1:,ii, 1].squeeze(), 'k')
    plt.title('with observation')
    x_hat=model.x_hat.cpu().detach().numpy().squeeze()[:,ii,:]
    z_x_hat=model.z_x_hat.cpu().detach().numpy().squeeze()[:,ii,:]
    z_x_hat=2*(z_x_hat-z_x_hat.min(axis=0))/(z_x_hat.max(axis=0)-z_x_hat.min(axis=0))-1
    plt.savefig(save_result_path + 'trj-'+str(ii)+'-maze-with-obsr.png')
    plt.savefig(save_result_path + 'trj-'+str(ii)+'-maze-with-obsr.svg', format='svg')


    mean_true=get_cdf(x_n[:,ii,:].mean(axis=1))
    mean_pred=get_cdf(x_hat.mean(axis=1))
    plt.figure()
    plt.plot(mean_true,mean_pred,'r')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('with observations')

    plt.savefig(save_result_path + 'trj-'+str(ii)+'-cdf-maze-with-obsr.png')
    plt.savefig(save_result_path + 'trj-'+str(ii)+'-cdf-maze-with-obsr.svg', format='svg')
    print('trj-%d'%(ii))
    print('LDIDPs-with-obsr-cc=%f,mse=%f,mae=%f,'%(get_metrics(z_tr[1:,ii,:], z_hat)))
    print('F-theta-with-obsr-cc=%f,mse=%f,mae=%f,'%(get_metrics(z_tr[:,ii,:], z_x_hat)))


# plt.figure()

for ii in trj_samples:
    z_hat = model(x,z, False)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:,ii, :]
    z_hat = 2 * (z_hat - z_hat.min(axis=0)) / (z_hat.max(axis=0) - z_hat.min(axis=0)) - 1
    f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
    axes[0].plot(z_hat[:, 0].squeeze(), 'b')
    axes[0].plot(z_tr[1:,ii, 0].squeeze(), 'k')

    axes[1].plot(z_hat[:, 1].squeeze(), 'b')
    axes[1].plot(z_tr[1:,ii, 1].squeeze(), 'k')
    plt.title('No observation')

    x_hat=model.x_hat.cpu().detach().numpy().squeeze()[:,ii,:]
    z_x_hat=model.z_x_hat.cpu().detach().numpy().squeeze()[:,ii,:]
    z_x_hat=2*(z_x_hat-z_x_hat.min(axis=0))/(z_x_hat.max(axis=0)-z_x_hat.min(axis=0))-1
    plt.savefig(save_result_path + 'trj-'+str(ii)+'-maze-with-no-obsr.png')
    plt.savefig(save_result_path + 'trj-'+str(ii)+'-maze-with-no-obsr.svg', format='svg')

    mean_true=get_cdf(x_n[:,ii,:].mean(axis=1))
    mean_pred=get_cdf(x_hat.mean(axis=1))
    plt.figure()
    plt.plot(mean_true,mean_pred, 'b')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('with-no-observations')

    plt.savefig(save_result_path + 'trj-'+str(ii)+'-cdf-maze-with-no-obsr.png')
    plt.savefig(save_result_path + 'trj-'+str(ii)+'-cdf-maze-with-no-obsr.svg', format='svg')
    print('trj-%d' % (ii))
    print('LDIDPs-no-obsr-cc=%f,mse=%f,mae=%f'%(get_metrics(z_tr[1:,ii,:], z_hat)))
    print('F-theta-no-obsr-cc=%f,mse=%f,mae=%f'%(get_metrics(z_tr[:,ii,:], z_x_hat)))
