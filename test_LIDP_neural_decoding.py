from scipy import io
from scipy import stats
import sys
import numpy as np
import pickle
import torch
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader

import LIDM
from LIDM import *
import torch
from torch.distributions import MultivariateNormal

folder='./NeuralData/' #ENTER THE FOLDER THAT YOUR DATA IS IN
# folder='/home/jglaser/Data/DecData/'
# folder='/Users/jig289/Dropbox/Public/Decoding_Data/'


with open(folder+'example_data_hc.pickle','rb') as f:
    neural_data,pos_binned=pickle.load(f,encoding='latin1') #If using python 3
#     neural_data,pos_binned=pickle.load(f)
bins_before=4 #How many bins of neural data prior to the output are used for decoding
bins_current=1 #Whether to use concurrent time bin of neural data
bins_after=5 #How many bins of neural data after the output are used for decoding
#Remove neurons with too few spikes in HC dataset
nd_sum=np.nansum(neural_data,axis=0) #Total number of spikes of each neuron
rmv_nrn=np.where(nd_sum<100) #Find neurons who have less than 100 spikes total
neural_data=np.delete(neural_data,rmv_nrn,1) #Remove those neurons
X=neural_data
#Set decoding output
y=pos_binned
#Number of bins to sum spikes over
N=bins_before+bins_current+bins_after
#Remove time bins with no output (y value)
rmv_time=np.where(np.isnan(y[:,0]) | np.isnan(y[:,1]))
X=np.delete(X,rmv_time,0)
y=np.delete(y,rmv_time,0)
training_range=[0, 0.5]
valid_range=[0.5,0.65]
testing_range=[0.65, 0.8]

#Number of examples after taking into account bins removed for lag alignment
num_examples=X.shape[0]

#Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
#This makes it so that the different sets don't include overlapping neural data
training_set=np.arange((np.round(training_range[0]*num_examples))+bins_before,(np.round(training_range[1]*num_examples))-bins_after)
testing_set=np.arange((np.round(testing_range[0]*num_examples))+bins_before,(np.round(testing_range[1]*num_examples))-bins_after)
valid_set=np.arange((np.round(valid_range[0]*num_examples))+bins_before,(np.round(valid_range[1]*num_examples))-bins_after)

#Get training data
X_train=X[training_set.astype('int'),:]
y_train=y[training_set.astype('int'),:]

#Get testing data
X_test=X[testing_set.astype('int'),:]
y_test=y[testing_set.astype('int'),:]

#Get validation data
X_valid=X[valid_set.astype('int'),:]
y_valid=y[valid_set.astype('int'),:]

downsample_rate=25
x_train_d=np.zeros((len(np.arange(0, X_train.shape[0], downsample_rate)),X_train.shape[1]))
x_test_d=np.zeros((len(np.arange(0, X_test.shape[0], downsample_rate)),X_test.shape[1]))
x_valid_d=np.zeros((len(np.arange(0, X_valid.shape[0], downsample_rate)),X_valid.shape[1]))
for ii in range(x_train_d.shape[1]):
        x_train_d[:, ii] = np.convolve(X_train[:, ii].squeeze(), np.ones((downsample_rate, 1)).squeeze(), 'same')[
            np.arange(0, X_train.shape[0], downsample_rate)]

        x_test_d[:, ii] = np.convolve(X_test[:, ii].squeeze(), np.ones((downsample_rate, 1)).squeeze(), 'same')[
            np.arange(0, X_test.shape[0], downsample_rate)]
        x_valid_d[:, ii] = np.convolve(X_valid[:, ii].squeeze(), np.ones((downsample_rate, 1)).squeeze(), 'same')[
            np.arange(0, X_valid.shape[0], downsample_rate)]

y_train_d=np.zeros((len(np.arange(0, y_train.shape[0], downsample_rate)),y_train.shape[1]))
y_test_d=np.zeros((len(np.arange(0, y_test.shape[0], downsample_rate)),y_test.shape[1]))
y_valid_d=np.zeros((len(np.arange(0, y_valid.shape[0], downsample_rate)),y_valid.shape[1]))
for ii in range(y_train.shape[1]):
        y_train_d[:, ii] = np.convolve(y_train[:, ii].squeeze(), np.ones((downsample_rate, 1)).squeeze(), 'same')[
            np.arange(0, y_train.shape[0], downsample_rate)]

        y_test_d[:, ii] = np.convolve(y_test[:, ii].squeeze(), np.ones((downsample_rate, 1)).squeeze(), 'same')[
            np.arange(0, y_test.shape[0], downsample_rate)]
        y_valid_d[:, ii] = np.convolve(y_valid[:, ii].squeeze(), np.ones((downsample_rate, 1)).squeeze(), 'same')[
            np.arange(0, y_valid.shape[0], downsample_rate)]
x=x_train_d
z=y_train_d
x = 2 * (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)) - 1
z = 2 * (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0)) - 1

device = torch.device('cuda')
Dataset = get_dataset(x, z, device)
Dataset_loader = DataLoader(Dataset, batch_size=x.shape[0], shuffle=False)
model = LIDM(latent_dim=z.shape[1], obser_dim=x.shape[1], sigma_x=.3, alpha=.1,importance_sample_size=1, n_layers=1,
              device=device).to(device)
model.apply(init_weights)
print(f'The g_theta model has {count_parameters(model.g_theta):,} trainable parameters')
print(f'The f_phi model has {count_parameters(model.f_phi):,} trainable parameters')
print(f'The f_phi.f_phi_x model has {count_parameters(model.f_phi.f_phi_x):,} trainable parameters')
print(f'The LIDM model has {count_parameters(model):,} trainable parameters')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
CLIP = 1
total_loss = []
Numb_Epochs = 400
for epoch in range(Numb_Epochs):
    epoch_loss = 0
    for i, batch in enumerate(Dataset_loader):
        x, z = batch
        x = torch.unsqueeze(x, 1)
        z = torch.unsqueeze(z, 1)

        optimizer.zero_grad()
        z_hat = model(x, True)
        print('epoch=%d/%d' % (epoch, Numb_Epochs))
        loss = model.loss(a=1, b=1)
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

plt.figure()
plt.plot(total_loss)
plt.show()

z = z.detach().cpu().numpy().squeeze()
# plt.figure()
f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
# ax = plt.axes(projection='3d')
trj_samples = np.random.randint(0, 10, 3)
for ii in trj_samples:
    z_hat = model(x, True)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:, ii, :]

    z_hat = 2 * (z_hat - z_hat.min(axis=0)) / (z_hat.max(axis=0) - z_hat.min(axis=0)) - 1

    axes[0].plot(z_hat[:, 0].squeeze(), 'r')
    axes[0].plot(z[1:, 0].squeeze(), 'k')

    axes[1].plot(z_hat[:, 1].squeeze(), 'r')
    axes[1].plot(z[1:, 1].squeeze(), 'k')
plt.title('with observation')
plt.show()

# plt.figure()
f, axes = plt.subplots(2, 1, sharex=True, sharey=False)

for ii in trj_samples:
    z_hat = model(x, False)
    z_hat = z_hat.detach().cpu().numpy().squeeze()[1:, ii, :]
    z_hat = 2 * (z_hat - z_hat.min(axis=0)) / (z_hat.max(axis=0) - z_hat.min(axis=0)) - 1
    axes[0].plot(z_hat[:, 0].squeeze(), 'b')
    axes[0].plot(z[1:, 0].squeeze(), 'k')

    axes[1].plot(z_hat[:, 1].squeeze(), 'b')
    axes[1].plot(z[1:, 1].squeeze(), 'k')
plt.title('No observation')
plt.show()