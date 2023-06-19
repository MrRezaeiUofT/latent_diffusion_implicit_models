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

class g_theta(nn.Module):
    def __init__(self, input_dim,
                 output_dim,
                 dropout):
        super().__init__()

        self.output_dim = output_dim
        self.input_dim=input_dim
        self.embedding = nn.Sequential(nn.Linear(input_dim, output_dim),
                                       nn.LeakyReLU(),
                                       nn.Linear(output_dim, output_dim),
                                       nn.Linear(output_dim, output_dim),
                                       nn.LeakyReLU(),
                                       nn.Linear(output_dim, output_dim),
                                       nn.LeakyReLU(),
                                       nn.Linear(output_dim, output_dim),
                                       # nn.Tanh()
                                       )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        outputs = self.dropout(self.embedding(src))

        return outputs

class f_phi_x(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, dropout ):
        super().__init__()

        self.output_dim = output_dim
        self.n_layers = n_layers
        self.input_dim=input_dim
        self.embedding =  nn.Sequential(nn.Linear(input_dim, input_dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(input_dim, input_dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(input_dim, input_dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(input_dim, input_dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(input_dim, output_dim),
                                        # nn.Tanh()
                                        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [src len, batch size, output_dim]
        return embedded

class f_phi(nn.Module):
    def __init__(self, obser_dim, latent_dim, n_layers,alpha, dropout, bidirectional,device):
        super().__init__()
        self.obser_dim = obser_dim
        self.device=device
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.alpha = alpha

        self.dropout=dropout
        self.f_phi_x= f_phi_x(input_dim=self.obser_dim,
                              output_dim=self.latent_dim,
                              n_layers=self.n_layers ,
                              dropout=self.dropout
                              )
        if bidirectional:
            self.fc_out = nn.Linear(2 * (latent_dim), (latent_dim))
        else:
            self.fc_out = nn.Linear((latent_dim), (latent_dim))

        self.embedding_z = nn.Linear((latent_dim), (latent_dim))
        self.rnn = nn.GRU((latent_dim), (latent_dim), n_layers, dropout=dropout, bidirectional =bidirectional )

        self.dropout = nn.Dropout(dropout)

    def forward(self, z_x_k,x_k, hidden, cell):

        # hidden=self.embedding_z(hidden)
        # hidden = hidden #+ self.sigma_z * torch.randn(hidden.shape).to(self.device)
        z_x_k=z_x_k.unsqueeze(0)
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

class LIDM(nn.Module):
    def __init__(self, latent_dim, obser_dim, sigma_x, alpha,importance_sample_size,n_layers, device):
        super().__init__()
        self.device = device
        self.latent_dim=torch.tensor([latent_dim], requires_grad=False).to(self.device)
        self.obser_dim =torch.tensor([obser_dim], requires_grad=False).to(self.device)

        self.sigma_x=torch.tensor([sigma_x], requires_grad=False).to(self.device)

        self.alpha = torch.tensor([alpha], requires_grad=False).to(self.device)
        self.importance_sample_size= importance_sample_size
        self.n_layers=n_layers
        self.dp_rate=.0
        self.f_phi = f_phi(obser_dim=self.obser_dim,
                          latent_dim=self.latent_dim,
                          n_layers= self.n_layers,
                           alpha=self.alpha,
                          dropout=self.dp_rate,
                          bidirectional=True,
                           device=self.device)

        self.g_theta = g_theta( input_dim=self.latent_dim,
                                output_dim=self.obser_dim,

                                dropout=self.dp_rate)


    def forward(self, obsrv, obsr_enable):
        # obsrv = [src len, 1, obsr dim]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        self.obsrv = obsrv#.repeat(1,self.importance_sample_size,1)
        batch_size = self.obsrv.shape[1]
        seq_len = self.obsrv.shape[0]

        # tensor to store decoder outputs
        self.z_hat = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        self.x_hat = torch.zeros(seq_len, batch_size, self.obser_dim).to(self.device)
        self.z_x_hat=torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        # outputs_neural = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder

        z= self.sigma_x*Variable(torch.randn((2*self.n_layers,batch_size, self.latent_dim))).to(self.device)
        self.z_hat[0] =  self.sigma_x*Variable(torch.randn(self.z_x_hat[0].shape)).to(self.device)
        cell=torch.zeros_like(z)
        output=self.z_hat[0]
        for k in range(1, seq_len):
            eps_z= self.alpha*self.sigma_x * torch.randn(self.z_hat[0].shape).to(self.device)
            eps_x=self.sigma_x * torch.randn( self.x_hat[0].shape).to(self.device)

            if obsr_enable:
                temp = torch.sqrt(1-torch.pow(self.sigma_x,2))*self.g_theta(output.clone())+eps_x #+
                self.x_hat[k]=temp
                self.z_x_hat[k] = torch.sqrt(1-torch.pow(self.alpha*self.sigma_x,2))*self.f_phi.f_phi_x(self.obsrv[k])+ eps_z
                output, z, cell = self.f_phi(
                    self.z_x_hat[k].clone(),temp, z, cell)
                self.z_hat[k]=output.clone()


            else:
                temp = torch.sqrt(1-torch.pow(self.sigma_x,2))*self.g_theta(output.clone())+eps_x
                self.x_hat[k]=temp.clone()
                self.z_x_hat[k] = torch.sqrt(1-torch.pow(self.alpha*self.sigma_x,2))*self.f_phi.f_phi_x(temp.clone())+  eps_z
                output, z, cell = self.f_phi(self.z_x_hat[k].clone(),temp.clone(), z, cell)
                self.z_hat[k]=output.clone()
        return self.z_hat


    def loss (self, a,b,c,z):
        L1=F.mse_loss(self.x_hat, self.obsrv)*(torch.pow(self.alpha,2))
        L2=F.mse_loss(torch.diff(self.z_hat,dim=0),
                      self.z_x_hat[1:])#*(torch.pow(self.alpha,2))#*torch.pow(self.sigma_z[1:],2)+1e-4)
        L3=F.mse_loss(z,self.z_hat)
        L= b*L2+ a*L1 + c*L3
        print('L1=%f, L2=%f, L3=%f, L=%f'%(L1,L2,L3,L))
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
