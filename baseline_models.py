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
class f_phi_RNN(nn.Module):
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

class RNN_baseline(nn.Module):
    def __init__(self, latent_dim, obser_dim, n_layers, device):
        super().__init__()
        self.device = device
        self.latent_dim=torch.tensor([latent_dim], requires_grad=False).to(self.device)
        self.obser_dim =torch.tensor([obser_dim], requires_grad=False).to(self.device)

        self.n_layers=n_layers
        self.dp_rate=.0
        self.f_phi = f_phi_RNN(obser_dim=self.obser_dim,
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

class encoder_seq2seq(nn.Module):
    def __init__(self, obser_dim, latent_dim, n_layers, dropout, bidirectional,device):
        super().__init__()
        self.obser_dim = obser_dim
        self.device=device
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional


        self.dropout=dropout

        self.embedding_z = nn.Sequential(
            nn.Linear(obser_dim, obser_dim),
                                       nn.LeakyReLU(),
                                       nn.Linear(obser_dim, latent_dim),
                                       nn.LeakyReLU(),
                                       nn.Linear(latent_dim, latent_dim),
                                       nn.Tanh()
                                       )
        self.rnn = nn.GRU((latent_dim), (latent_dim), n_layers, dropout=dropout, bidirectional =bidirectional )

        self.dropout = nn.Dropout(dropout)

    def forward(self, z_x_k):


        # hidden = hidden #+ self.sigma_z * torch.randn(hidden.shape).to(self.device)
        # z_x_k=z_x_k.unsqueeze(0)
        z_x_k = self.embedding_z(z_x_k)
        # x_k = x_k.unsqueeze(0)
        # embedded_new=torch.concat([z_x_k,x_k], dim=-1)
        # embedded = [1, batch size, latent dim]
        # embedded_new= embedded_new#+self.sigma_x * torch.randn(embedded_new.shape).to(self.device)

        # embedded_new = self.dropout(self.embedding_z(embedded_new))
        output, hidden = self.rnn(z_x_k)

        # output = [seq len, batch size, out dim * n directions]
        # hidden = [n layers * n directions, batch size, out dim]
        # cell = [n layers * n directions, batch size, out dim]


        # prediction=torch.sqrt(self.alpha)*embedded_new+ torch.sqrt(1-self.alpha)*prediction
        # prediction = [batch size, output dim]

        return output, hidden

class decoder_seq2seq(nn.Module):
    def __init__(self, obser_dim, latent_dim, n_layers, dropout, bidirectional,device):
        super().__init__()
        self.obser_dim = obser_dim
        self.device=device
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional


        self.dropout=dropout

        if bidirectional:
            self.fc_out = nn.Linear(2 * (obser_dim), (latent_dim))
            self.embedding_z = nn.Sequential(
                nn.Linear(2*obser_dim, obser_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(obser_dim, obser_dim),
                                             nn.LeakyReLU(),

                                             )
        else:
            self.fc_out = nn.Linear((obser_dim), (latent_dim))
            self.embedding_z = nn.Sequential(
                nn.Linear(obser_dim, obser_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(obser_dim, obser_dim),
                                             nn.LeakyReLU(),

                                             )

        
        self.rnn = nn.GRU((obser_dim), (obser_dim), n_layers, dropout=dropout, bidirectional =bidirectional )

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

        return prediction,output, hidden, cell

class VariationalSeq2Seq_baseline(nn.Module):
    def __init__(self, latent_dim, obser_dim, n_layers, device):
        super().__init__()
        self.device = device
        self.latent_dim=torch.tensor([latent_dim], requires_grad=False).to(self.device)
        self.obser_dim =torch.tensor([obser_dim], requires_grad=False).to(self.device)

        self.n_layers=n_layers
        self.dp_rate=.0
        self.bidirectional=True
        self.hidden_dim=10
        self.f_phi_Enc = encoder_seq2seq(obser_dim=self.obser_dim,
                          latent_dim=self.hidden_dim,
                          n_layers= self.n_layers,

                          dropout=self.dp_rate,
                          bidirectional=self.bidirectional,
                           device=self.device)
        self.f_phi_Dec = decoder_seq2seq(obser_dim=self.hidden_dim,
                              latent_dim=self.latent_dim,
                              n_layers=self.n_layers,

                              dropout=self.dp_rate,
                              bidirectional=self.bidirectional,
                              device=self.device)
        # if self.bidirectional:
        #     self.context_to_mu = nn.Linear(self.f_phi_Enc.latent_dim*2, self.f_phi_Enc.latent_dim)
        #     self.context_to_logvar = nn.Linear(self.f_phi_Enc.latent_dim*2, self.f_phi_Enc.latent_dim)
        # else:
        self.context_to_mu = nn.Linear(self.f_phi_Enc.latent_dim, self.f_phi_Enc.latent_dim)
        self.context_to_logvar = nn.Linear(self.f_phi_Enc.latent_dim, self.f_phi_Enc.latent_dim)
    def forward(self, obsrv):
        # obsrv = [src len, 1, obsr dim]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        self.obsrv = obsrv#.repeat(1,self.importance_sample_size,1)
        batch_size = self.obsrv.shape[1]
        seq_len = self.obsrv.shape[0]

        # tensor to store decoder outputs
        self.z_hat = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        _, hidden = self.f_phi_Enc( obsrv)
        z = Variable(torch.randn(hidden.shape)).to(self.device)
        cell=torch.zeros_like(z)
        mu = self.context_to_mu(hidden)
        logvar = self.context_to_logvar(hidden)
        std = torch.exp(0.5 * logvar)
        z =hidden# z * std + mu
        output=  torch.zeros( batch_size, 2*self.hidden_dim).to(self.device)
        self.kld =0# (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()
        for k in range(1, seq_len):

            prediction,output, z, cell = self.f_phi_Dec(
                    output.squeeze().clone(), z.squeeze(), cell)
            self.z_hat[k]=prediction
        return self.z_hat


    def loss (self, z):

        L1=F.mse_loss(z,self.z_hat)
        L2=self.kld
        print(' L1=%f, L2=%f'%(L1,L2))
        return L1+1*L2


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
