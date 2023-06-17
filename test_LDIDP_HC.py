import pickle

import numpy as np
import torch
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader
from utils import *
import LIDM
from LIDM import *
dataset = pickle.load(open("./NeuralData/example_data_hc_mine.p", "rb"))
batch_size=2
''' data'''


def preprocess_HC(x_in,z_in,downsample_rate, episod_len):
    x = calSmoothNeuralActivity(np.squeeze(x_in), 40, 8)
    z = z_in
    data_len = x.shape[0]

    ''' normalization'''
    x = 2 * (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)) - 1
    z = 2 * (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0)) - 1

    ''' down sampling rate'''

    x = x[np.arange(0, data_len, downsample_rate), :]
    x = calDesignMatrix_V2(x, 4).squeeze()
    z = z[np.arange(0, data_len, downsample_rate), :]

    x_new = np.zeros((episod_len,(x.shape[0]//episod_len), x.shape[1]))
    z_new = np.zeros((episod_len, (x.shape[0]//episod_len), z.shape[1]))
    for ii in range(x.shape[0]//episod_len):
        x_new[:, ii, :] = x[ii*episod_len:(ii+1)*episod_len, :]
        z_new[:, ii, :] = z[ii*episod_len:(ii+1)*episod_len, :]
    return x_new,z_new
[x_tr,z_tr]= preprocess_HC(dataset['X_trian'],dataset['Y_trian'][:, :2], 1,1000)
[x_val,z_val]= preprocess_HC(dataset['X_valid'],dataset['Y_valid'][:, :2], 1,4000)
''''''
device = torch.device('cuda')
Dataset = get_dataset_HC(x_tr, z_tr, device)
Dataset_val = get_dataset_HC(x_val, z_val, device)
Dataset_loader = DataLoader(Dataset, batch_size=batch_size,shuffle=False)
Dataset_val_loader = DataLoader(Dataset_val, batch_size=z_val.shape[1],shuffle=False)
model = LIDM(latent_dim=z_tr.shape[-1], obser_dim=x_tr.shape[-1], sigma_x=.2,alpha=.1, importance_sample_size=1, n_layers=5,
              device=device).to(device)
model.apply(init_weights)
print(f'The g_theta model has {count_parameters(model.g_theta):,} trainable parameters')
print(f'The f_phi model has {count_parameters(model.f_phi):,} trainable parameters')
print(f'The f_phi.f_phi_x model has {count_parameters(model.f_phi.f_phi_x):,} trainable parameters')
print(f'The LIDM model has {count_parameters(model):,} trainable parameters')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
CLIP = 1
total_loss=[]
Numb_Epochs=1000
for epoch in range(Numb_Epochs):
    epoch_loss = 0
    for i, batch in enumerate(Dataset_loader):
        x, z = batch
        x= torch.swapaxes(x, 0,1)
        z = torch.swapaxes(z, 0,1)

        optimizer.zero_grad()
        z_hat = model(x,True)
        print('epoch=%d/%d'%(epoch,Numb_Epochs))
        loss=model.loss(a=1,b=1,c=10,z=z)
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

''' save and load models'''
torch.save(model.state_dict(), 'final_model_HC_1mc_1000.pt')

# model.load_state_dict(torch.load('final_model_HC.pt'))
''''''

import matplotlib.pyplot as plt

''' visualization'''


save_result_path = 'Results/'
plt.figure()
plt.plot(total_loss)
plt.show()

Dataset_loader = DataLoader(Dataset, batch_size=z_tr.shape[1],shuffle=False)
for i, batch in enumerate(Dataset_loader):
    x, z = batch
    x = torch.swapaxes(x, 0, 1)
    z = torch.swapaxes(z, 0, 1)

z = z.detach().cpu().numpy().squeeze()
# plt.figure()

# plt.figure()
# x_n = x.detach().cpu().numpy().squeeze()
# plt.imshow(x_n[:,-1,:].T)
# plt.show()

trj_samples = np.arange(0, z.shape[1])
for ii in trj_samples:
    z_hat = model(x, True)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:,ii, :]

    z_hat = 2 * (z_hat - z_hat.min(axis=0)) / (z_hat.max(axis=0) - z_hat.min(axis=0)) - 1

    f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
    axes[0].plot(z_hat[:, 0].squeeze(), 'r')
    axes[0].plot(z[1:,ii, 0].squeeze(), 'k')

    axes[1].plot(z_hat[:, 1].squeeze(), 'r')
    axes[1].plot(z[1:,ii, 1].squeeze(), 'k')


    plt.title('with observations')
    plt.savefig(save_result_path + 'Train-'+str(ii)+'-HC-with-obsr.png')
    plt.savefig(save_result_path + 'Train-'+str(ii)+'-HC-with-obsr.svg', format='svg')
    # plt.colse()


for ii in trj_samples:
    f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
    z_hat = model(x, False)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:,ii, :]
    z_hat = 2 * (z_hat - z_hat.min(axis=0)) / (z_hat.max(axis=0) - z_hat.min(axis=0)) - 1
    axes[0].plot(z_hat[:, 0].squeeze(), 'b')
    axes[0].plot(z[1:,ii, 0].squeeze(), 'k')

    axes[1].plot(z_hat[:, 1].squeeze(), 'b')
    axes[1].plot(z[1:,ii, 1].squeeze(), 'k')
    plt.title('No observation')

    plt.savefig(save_result_path + 'Train-'+str(ii)+'-HC-without-obsr.png')
    plt.savefig(save_result_path + 'Train-'+str(ii)+'-HC-without-obsr.svg', format='svg')
    # plt.close()


""" validation result """
for i, batch in enumerate(Dataset_val_loader):
    x, z = batch
    x = torch.swapaxes(x, 0, 1)
z = z.detach().cpu().numpy().squeeze()
# plt.figure()

# ff=plt.figure()
# ax = ff.axes(projection='3d')
trj_samples = np.arange(z_val.shape[1])
for ii in trj_samples:
    f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
    z_hat = model(x, True)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:, :]

    z_hat = 2 * (z_hat - z_hat.min(axis=0)) / (z_hat.max(axis=0) - z_hat.min(axis=0)) - 1

    axes[0].plot(z_hat[:, 0].squeeze(), 'r')
    axes[0].plot(z[1:, 0].squeeze(), 'k')

    axes[1].plot(z_hat[:, 1].squeeze(), 'r')
    axes[1].plot(z[1:, 1].squeeze(), 'k')


    plt.title('with observations')
    plt.savefig(save_result_path + 'Test-'+str(ii)+'-HC-with-obsr.png')
    plt.savefig(save_result_path + 'Test-'+str(ii)+'-HC-with-obsr.svg', format='svg')
    # plt.close()


# print('Test-LDIDPs-with-obsr-cc=%f,mse=%f,mae=%f' % (get_metrics(z[1:], z_hat)))


# plt.figure()


for ii in trj_samples:
    f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
    z_hat = model(x, False)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:,:]
    z_hat = 2 * (z_hat - z_hat.min(axis=0)) / (z_hat.max(axis=0) - z_hat.min(axis=0)) - 1
    axes[0].plot(z_hat[:, 0].squeeze(), 'b')
    axes[0].plot(z[1:, 0].squeeze(), 'k')

    axes[1].plot(z_hat[:, 1].squeeze(), 'b')
    axes[1].plot(z[1:, 1].squeeze(), 'k')
    plt.title('No observation')
    plt.savefig(save_result_path + 'Test-'+str(ii)+'-HC-without-obsr.png')
    plt.savefig(save_result_path + 'Test-'+str(ii)+'-HC-without-obsr.svg', format='svg')
    # plt.close()
# print('Test:LDIDPs-without-obsr-cc=%f,mse=%f,mae=%f' % (get_metrics(z[1:], z_hat)))


