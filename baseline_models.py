import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
SEED = 12
from torch.autograd import Variable
random.seed(SEED)
np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True




class f_phi(nn.Module):
    def __init__(self, obser_dim, latent_dim, n_layers, dropout, bidirectional,device):
        super().__init__()
        self.obser_dim = obser_dim
        self.device=device
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional


        self.dropout=dropout

        if bidirectional:
            self.fc_out = nn.Linear(2 * (latent_dim), (latent_dim))
        else:
            self.fc_out = nn.Linear((latent_dim), (latent_dim))

        self.embedding_z = nn.Sequential(nn.Linear(obser_dim, obser_dim),
                                       nn.LeakyReLU(),
                                       nn.Linear(obser_dim, latent_dim),
                                       nn.LeakyReLU(),
                                       nn.Linear(latent_dim, latent_dim),

                                       # nn.Tanh()
                                       )
        self.rnn = nn.GRU((latent_dim), (latent_dim), n_layers, dropout=dropout, bidirectional =bidirectional )

        self.dropout = nn.Dropout(dropout)

    def forward(self, z_x_k, hidden, cell):


        # hidden = hidden #+ self.sigma_z * torch.randn(hidden.shape).to(self.device)
        z_x_k=z_x_k.unsqueeze(0)
        z_x_k = self.embedding_z(z_x_k)
        # x_k = x_k.unsqueeze(0)
        # embedded_new=torch.concat([z_x_k,x_k], dim=-1)
        # embedded = [1, batch size, latent dim]
        # embedded_new= embedded_new#+self.sigma_x * torch.randn(embedded_new.shape).to(self.device)

        # embedded_new = self.dropout(self.embedding_z(embedded_new))
        output, hidden = self.rnn(z_x_k, hidden)

        # output = [seq len, batch size, out dim * n directions]
        # hidden = [n layers * n directions, batch size, out dim]
        # cell = [n layers * n directions, batch size, out dim]

        prediction = self.fc_out(output.squeeze(0))
        # prediction=torch.sqrt(self.alpha)*embedded_new+ torch.sqrt(1-self.alpha)*prediction
        # prediction = [batch size, output dim]

        return prediction, hidden, cell

class GRU_baseline(nn.Module):
    def __init__(self, latent_dim, obser_dim, n_layers, device):
        super().__init__()
        self.device = device
        self.latent_dim=torch.tensor([latent_dim], requires_grad=False).to(self.device)
        self.obser_dim =torch.tensor([obser_dim], requires_grad=False).to(self.device)

        self.n_layers=n_layers
        self.dp_rate=.0
        self.f_phi = f_phi(obser_dim=self.obser_dim,
                          latent_dim=self.latent_dim,
                          n_layers= self.n_layers,

                          dropout=self.dp_rate,
                          bidirectional=True,
                           device=self.device)


    def forward(self, obsrv):
        # obsrv = [src len, 1, obsr dim]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        self.obsrv = obsrv#.repeat(1,self.importance_sample_size,1)
        batch_size = self.obsrv.shape[1]
        seq_len = self.obsrv.shape[0]

        # tensor to store decoder outputs
        self.z_hat = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)

        # outputs_neural = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder

        z= 0*Variable(torch.randn((2*self.n_layers,batch_size, self.latent_dim))).to(self.device)
        self.z_hat[0] =  0*Variable(torch.randn(self.z_hat[0].shape)).to(self.device)
        cell=torch.zeros_like(z)

        for k in range(1, seq_len):




            output, z, cell = self.f_phi(
                    obsrv[k], z, cell)
            self.z_hat[k]=output.clone()



        return self.z_hat


    def loss (self, z):

        L=F.mse_loss(z,self.z_hat)

        print(' L=%f'%(L))
        return L



class VRNN_baseline(nn.Module):
    def __init__(self, latent_dim, obser_dim, n_layers, device):
        super().__init__()
        self.device = device
        self.latent_dim=torch.tensor([latent_dim], requires_grad=False).to(self.device)
        self.obser_dim =torch.tensor([obser_dim], requires_grad=False).to(self.device)

        self.n_layers=n_layers
        self.dp_rate=.0
        self.biodirectional=True
        self.f_phi_mu = f_phi(obser_dim=self.obser_dim,
                          latent_dim=self.latent_dim,
                          n_layers= self.n_layers,

                          dropout=self.dp_rate,
                          bidirectional=self.biodirectional,
                           device=self.device)
        self.f_phi_std = f_phi(obser_dim=self.obser_dim,
                              latent_dim=self.latent_dim,
                              n_layers=self.n_layers,

                              dropout=self.dp_rate,
                              bidirectional=self.biodirectional,
                              device=self.device)


    def forward(self, obsrv):
        # obsrv = [src len, 1, obsr dim]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        self.obsrv = obsrv#.repeat(1,self.importance_sample_size,1)
        batch_size = self.obsrv.shape[1]
        seq_len = self.obsrv.shape[0]

        # tensor to store decoder outputs
        self.z_hat = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)

        # outputs_neural = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder

        z_mu= 0*Variable(torch.randn((2*self.n_layers,batch_size, self.latent_dim))).to(self.device)
        cell_mu=torch.zeros_like(z_mu)
        z_logvar = 0 * Variable(torch.randn((2 * self.n_layers, batch_size, self.latent_dim))).to(self.device)
        cell_logvar = torch.zeros_like(z_mu)
        self.kld=0
        for k in range(1, seq_len):

            mu_k, z_mu, cell_mu = self.f_phi_mu(
                    obsrv[k], z_mu, cell_mu)
            logvar_k, z_logvar, cell_logvar = self.f_phi_std(
                obsrv[k], z_logvar, cell_logvar)

            std_k = torch.exp(0.5 * logvar_k)
            output = Variable(torch.randn(self.z_hat[k].shape)).to(self.device) * std_k + mu_k
            self.kld += (-0.5 * torch.sum(logvar_k - torch.pow(mu_k, 2) - torch.exp(logvar_k) + 1, 1)).mean().squeeze()
            self.z_hat[k]=output



        return self.z_hat


    def loss (self, z):

        L1=F.mse_loss(z,self.z_hat)
        L2=self.kld
        print(' L1=%f, L2=%f'%(L1,L2))
        return L1+L2


class get_dataset_HC(Dataset):

    ''' create a dataset suitable for pytorch models'''
    def __init__(self, x,z, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.z = torch.tensor(z, dtype=torch.float32).to(device)

    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, index):
        return [self.x[:,index,:], self.z[:,index,:]]

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.5, 0.5)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
