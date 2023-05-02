import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
SEED = 1234
from torch.autograd import Variable
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class g_theta(nn.Module):
    def __init__(self, input_dim,
                 output_dim,
                 n_layers,
                 dropout):
        super().__init__()

        self.output_dim = output_dim

        self.input_dim=input_dim
        self.embedding = nn.Sequential(nn.Linear(input_dim, input_dim),
                                       nn.Linear(input_dim, input_dim),
                                       nn.Linear(input_dim, output_dim)
                                       )
        # self.rnn = nn.LSTM(input_dim, input_dim, n_layers, dropout=dropout, bidirectional =bidirectional )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        outputs = self.dropout(self.embedding(src))
        # outputs = [src len, batch size, observation dim]
        return outputs

class f_phi_x(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, dropout ):
        super().__init__()

        self.output_dim = output_dim
        self.n_layers = n_layers
        self.input_dim=input_dim
        self.embedding =  nn.Sequential(nn.Linear(input_dim, output_dim),
                                        nn.Linear(output_dim, output_dim),
                                        nn.Linear(output_dim, output_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [src len, batch size, output_dim]
        return embedded

class f_phi(nn.Module):
    def __init__(self, obser_dim, latent_dim, n_layers,alpha, sigma_x, sigma_z, dropout, bidirectional ):
        super().__init__()
        self.obser_dim = obser_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.alpha = alpha
        self.sigma_x=sigma_x
        self.sigma_z=sigma_z
        self.dropout=dropout
        self.f_phi_x= f_phi_x(input_dim=self.obser_dim,
                              output_dim=self.latent_dim,
                              n_layers=self.n_layers ,
                              dropout=self.dropout
                              )
        if bidirectional:
            self.fc_out = nn.Linear(2 * latent_dim, latent_dim)
        else:
            self.fc_out = nn.Linear(latent_dim, latent_dim)

        self.embedding_z = nn.Linear(latent_dim, latent_dim)
        self.rnn = nn.LSTM(latent_dim, latent_dim, n_layers, dropout=dropout, bidirectional =bidirectional )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_k, hidden, cell):

        hidden=self.embedding_z(hidden)
        hidden = hidden + self.sigma_z * torch.randn(hidden.shape)
        embedded_new=self.f_phi_x(x_k)
        # embedded = [1, batch size, latent dim]
        embedded_new= torch.sqrt(self.alpha)*embedded_new+self.sigma_x * torch.randn(embedded_new.shape) # + torch.sqrt(1-self.alpha)*hidden
        embedded_new=embedded_new.unsqueeze(0)
        # embedded_new = self.dropout(self.embedding_z(embedded_new))
        output, (hidden, cell) = self.rnn(embedded_new, (hidden,cell))

        # output = [seq len, batch size, out dim * n directions]
        # hidden = [n layers * n directions, batch size, out dim]
        # cell = [n layers * n directions, batch size, out dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell

class LIDM(nn.Module):
    def __init__(self, latent_dim, obser_dim, sigma_x, sigma_z, alpha, device):
        super().__init__()
        self.latent_dim=latent_dim
        self.obser_dim =obser_dim
        self.device=device
        self.sigma_x=torch.tensor([sigma_x], requires_grad=False )
        self.sigma_z=torch.tensor([sigma_z], requires_grad=False )
        self.alpha = torch.tensor([alpha], requires_grad=False)
        self.n_layers=5
        self.dp_rate=.1
        self.f_phi = f_phi(obser_dim=self.obser_dim,
                          latent_dim=self.latent_dim,
                          n_layers= self.n_layers,
                           alpha=self.alpha,
                           sigma_x=self.sigma_x,
                           sigma_z=self.sigma_z,
                          dropout=self.dp_rate,
                          bidirectional=False)

        self.g_theta = g_theta( input_dim=self.latent_dim,
                                output_dim=self.obser_dim,
                                n_layers= self.n_layers,
                                dropout=self.dp_rate)


    def forward(self, obsrv):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        self.obsrv = obsrv
        batch_size = self.obsrv.shape[1]
        seq_len = self.obsrv.shape[0]

        # tensor to store decoder outputs
        self.outputs = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        # outputs_neural = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder

        z= Variable(torch.randn((self.n_layers,batch_size, self.latent_dim))).to(self.device)
        cell=z
        for k in range(1, seq_len):

            output, z, cell = self.f_phi(self.obsrv[k], z, cell)

            # place predictions in a tensor holding predictions for each token
            self.outputs[k] = output

        return self.outputs
    def loss (self):
        L1=F.mse_loss(self.obsrv[1:], self.g_theta(self.outputs[:-1]))
        L2=F.mse_loss(torch.sqrt(self.alpha)*self.f_phi.f_phi_x(self.obsrv[1:]),
                      self.outputs[1:]-torch.sqrt(1-self.alpha)*self.outputs[:-1])

        L= L2+ L1*torch.pow(self.sigma_x,2)/(torch.pow(self.sigma_z,2)*torch.sqrt(self.alpha))
        return L



class get_dataset(Dataset):

    ''' create a dataset suitable for pytorch models'''
    def __init__(self, x,z, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.z = torch.tensor(z, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return [self.x[index], self.z[index]]


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.8, 0.8)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)