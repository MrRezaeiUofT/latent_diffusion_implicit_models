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

''' data'''
x = calSmoothNeuralActivity(np.squeeze(dataset['X_test']),100,8)
z = dataset['Y_test'][:,:2]
data_len=x.shape[0]

''' normalization'''
x = 2*(x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))-1
z = 2*(z-z.min(axis=0))/(z.max(axis=0)-z.min(axis=0))-1

''' down sampling rate'''
downsample_rate=10
x=x[np.arange(0, data_len, downsample_rate),:]
x = calDesignMatrix_V2(x,1).squeeze()
z=z[np.arange(0, data_len, downsample_rate),:]

''''''
device = torch.device('cuda')
Dataset = get_dataset(x, z, device)
Dataset_loader = DataLoader(Dataset, batch_size=x.shape[0],shuffle=False)
model = LIDM(latent_dim=z.shape[1], obser_dim=x.shape[1], sigma_x=.1,alpha=.2, importance_sample_size=1, n_layers=2,
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
plt.imshow(x.cpu().detach().numpy().squeeze().T)
plt.show()
x_n = x.detach().cpu().numpy().squeeze()
save_result_path = 'Results/'
plt.figure()
plt.plot(total_loss)
plt.show()

z = z.detach().cpu().numpy().squeeze()
# plt.figure()
f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
# ff=plt.figure()
# ax = ff.axes(projection='3d')
trj_samples = np.random.randint(0, 1, 1)
for ii in trj_samples:
    z_hat = model(x, True)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:, :]

    z_hat = 2 * (z_hat - z_hat.min(axis=0)) / (z_hat.max(axis=0) - z_hat.min(axis=0)) - 1

    axes[0].plot(z_hat[:, 0].squeeze(), 'r')
    axes[0].plot(z[1:, 0].squeeze(), 'k')

    axes[1].plot(z_hat[:, 1].squeeze(), 'r')
    axes[1].plot(z[1:, 1].squeeze(), 'k')

    # axes[2].plot(z_hat[:, 2].squeeze(), 'r')
    # axes[2].plot(z[1:, 2].squeeze(), 'k')
    # ax.plot3D(z[1:, 0].squeeze(), z[1:, 1].squeeze(), z[1:, 2].squeeze(), 'gray')
    # ax.plot3D(z_hat[:, 0].squeeze(), z_hat[:, 1].squeeze(), z_hat[:, 2].squeeze(), 'r')
plt.title('with observations')
x_hat = model.x_hat.cpu().detach().numpy().squeeze()

z_x_hat = model.z_x_hat.cpu().detach().numpy().squeeze()
z_x_hat = 2 * (z_x_hat - z_x_hat.min(axis=0)) / (z_x_hat.max(axis=0) - z_x_hat.min(axis=0)) - 1

plt.savefig(save_result_path + 'HC-with-obsr.png')
plt.savefig(save_result_path + 'HC-with-obsr.svg', format='svg')
plt.show()

mean_true = get_cdf(x_n.mean(axis=1))
mean_pred = get_cdf(x_hat.mean(axis=1))
plt.figure()
plt.plot(mean_true, mean_pred, 'r')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('with observations')
plt.show()
plt.savefig(save_result_path + 'cdf-HC-with-obsr.png')
plt.savefig(save_result_path + 'cdf-HC-with-obsr.svg', format='svg')
print('LDIDPs-with-obsr-cc=%f,mse=%f,mae=%f' % (get_metrics(z[1:], z_hat)))
print('F-theta-with-obsr-cc=%f,mse=%f,mae=%f' % (get_metrics(z, z_x_hat)))

# plt.figure()
f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
# ff= plt.figure()
# ax = ff.axes(projection='3d')
for ii in trj_samples:
    z_hat = model(x, False)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:, :]
    z_hat = 2 * (z_hat - z_hat.min(axis=0)) / (z_hat.max(axis=0) - z_hat.min(axis=0)) - 1
    axes[0].plot(z_hat[:, 0].squeeze(), 'b')
    axes[0].plot(z[1:, 0].squeeze(), 'k')

    axes[1].plot(z_hat[:, 1].squeeze(), 'b')
    axes[1].plot(z[1:, 1].squeeze(), 'k')

    # axes[2].plot(z_hat[:, 2].squeeze(), 'b')
    # axes[2].plot(z[1:, 2].squeeze(), 'k')
    # ax.plot3D(z[1:, 0].squeeze(), z[1:, 1].squeeze(), z[1:, 2].squeeze(), 'gray')
    # ax.plot3D(z_hat[:, 0].squeeze(), z_hat[:, 1].squeeze(), z_hat[:, 2].squeeze(), 'b')
plt.title('No observation')
x_hat = model.x_hat.cpu().detach().numpy().squeeze()

z_x_hat = model.z_x_hat.cpu().detach().numpy().squeeze()
z_x_hat = 2 * (z_x_hat - z_x_hat.min(axis=0)) / (z_x_hat.max(axis=0) - z_x_hat.min(axis=0)) - 1

plt.savefig(save_result_path + 'HC-without-obsr.png')
plt.savefig(save_result_path + 'HC-without-obsr.svg', format='svg')
plt.show()

mean_true = get_cdf(x_n.mean(axis=1))
mean_pred = get_cdf(x_hat.mean(axis=1))
plt.figure()
plt.plot(mean_true, mean_pred, 'b')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('without observations')
plt.show()
plt.savefig(save_result_path + 'cdf-HC-without-obsr.png')
plt.savefig(save_result_path + 'cdf-HC-without-obsr.svg', format='svg')
print('LDIDPs-without-obsr-cc=%f,mse=%f,mae=%f' % (get_metrics(z[1:], z_hat)))
print('F-theta-without-obsr-cc=%f,mse=%f,mae=%f' % (get_metrics(z, z_x_hat)))
