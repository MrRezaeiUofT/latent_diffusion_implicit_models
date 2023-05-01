import torch
import torch.nn as nn
import numpy as np

import random

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class g_theta(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, dropout, bidirectional =False ):
        super().__init__()

        self.output_dim = output_dim
        self.n_layers = n_layers
        self.input_dim=input_dim
        self.embedding = nn.Sequential(nn.Linear(input_dim, input_dim),nn.Linear(input_dim, output_dim))
        self.rnn = nn.LSTM(input_dim, input_dim, n_layers, dropout=dropout, bidirectional =bidirectional )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        # embedded = self.mixer(embedded)
        # embedded = [src len, batch size, out dim]

        # embedded=torch.concatenate((embedded,src[:,:,:1]), axis=-1)
        outputs, (hidden,cell) = self.rnn(src)

        # outputs = [src len, batch size, input dim * n directions]
        # hidden = [n layers * n directions, batch size, input dim]
        # cell = [n layers * n directions, batch size, input dim]

        # outputs are always from the top hidden layer
        outputs = self.dropout(self.embedding(outputs))
        return hidden, cell, outputs

class f_phi_x(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, dropout ):
        super().__init__()

        self.output_dim = output_dim
        self.n_layers = n_layers
        self.input_dim=input_dim
        self.embedding =  nn.Sequential(nn.Linear(input_dim, output_dim),nn.Linear(output_dim, output_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [src len, batch size, output_dim]
        return embedded

class f_phi(nn.Module):
    def __init__(self, obser_dim, latent_dim, n_layers, dropout, bidirectional ):
        super().__init__()
        self.obser_dim = obser_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.alpha = .5
        self.f_phi_x= f_phi_x(input_dim=obser_dim,
                              output_dim=latent_dim,
                              n_layers=2,
                              dropout=.1
                              )
        if bidirectional:
            self.fc_out = nn.Linear(2 * latent_dim, latent_dim)
        else:
            self.fc_out = nn.Linear(latent_dim, latent_dim)

        self.embedding_z = nn.Linear(latent_dim, latent_dim)
        self.rnn = nn.LSTM(latent_dim, latent_dim, n_layers, dropout=dropout, bidirectional =bidirectional )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_k, hidden, cell):

        embedded = self.dropout(self.embedding_z(hidden))

        # embedded = [1, batch size, latent dim]
        embedded_new= torch.sqrt(self.alpha)*self.f_phi_x(x_k) + torch.sqrt(1-self.alpha)*embedded
        output, (hidden, cell) = self.rnn(embedded_new, (hidden,cell))

        # output = [seq len, batch size, out dim * n directions]
        # hidden = [n layers * n directions, batch size, out dim]
        # cell = [n layers * n directions, batch size, out dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell

class LIDM(nn.Module):
    def __init__(self, latent_dim, obser_dim, device):
        super().__init__()
        self.latent_dim=latent_dim
        self.obser_dim =obser_dim
        self.device=device
        self.f_phi = f_phi(obser_dim=self.obser_dim,
                          latent_dim=self.latent_dim,
                          n_layers=2,
                          dropout=.1,
                          bidirectional=False)

        self.g_theta = g_theta( input_dim=self.latent_dim,
                                output_dim=self.obser_dim,
                                n_layers=2,
                                dropout=.1,
                                bidirectional =False)


    def forward(self, obsrv):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = obsrv.shape[1]
        seq_len = obsrv.shape[0]

        # tensor to store decoder outputs
        self.outputs = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        # outputs_neural = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder

        z = torch.distributions.MultivariateNormal(np.zeros(self.latent_dim), np.eye(self.latent_dim)).sample((1,))

        for k in range(1, seq_len):

            output, z, cell = self.f_phi(obsrv[k], z, cell)

            # place predictions in a tensor holding predictions for each token
            self.outputs[k] = output

        return self.outputs
    def loss (self, observations):






def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
