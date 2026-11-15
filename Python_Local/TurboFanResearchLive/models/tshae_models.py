import torch
import torch.nn as nn
import numpy as np



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, dropout_lstm, dropout=0, num_layers=1, bidirectional=True):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.p_lstm = dropout_lstm
        self.p = dropout

        self.lstm = nn.LSTM( #encoder part, 21 sensors
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.p_lstm,
            batch_first=True,
            bidirectional=self.bidirectional #run it through a bidircetional LSTM to read data forward and backward
        )

        self.fc_mean = nn.Sequential( #one linear layer that compresses the output of the LSTM to a latent dimension
            nn.Dropout(self.p),
            nn.Linear(
                in_features=self.num_directions * hidden_size,
                out_features=latent_dim)
        )

        self.fc_log_var = nn.Sequential( #another linear layer
            nn.Dropout(self.p),
            nn.Linear(
                in_features=self.num_directions * hidden_size,
                out_features=latent_dim)
        )

    def reparameterization(self, mean, var): #adds noise to the mean
        epsilon = torch.randn_like(var).to(var.device)
        z = mean + var * epsilon
        return z #z is the latent vector, mean is teh true center of where the encoder thinks the engine state is in the latent space
    #log_var is how uncertain encoder is

    def forward(self, x): #
        batch_size = x.shape[0]
        _, (h_n, _) = self.lstm(x) #run through LSTM
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        """
        hidden.shape = (num_layers*num_directions, batch, hidden_size)
        layers can be separated using h_n.view(num_layers, num_directions, batch, hidden_size)
        So you shouldn’t simply do hidden[-1] but first do a view() to separate the num_layers and num_directions (1 or 2). If you do

        hidden = hidden.view(num_layers, 2, batch, hidden_size) # 2 for bidirectional
        last_hidden = hidden[-1]
        then last_hidden.shape = (2, batch, hidden_size) and you can do

        last_hidden_fwd = last_hidden[0]
        last_hidden_bwd = last_hidden[1]

        TODO: check if it same as
        # Pass the input through the LSTM
        output, (h_n, c_n) = lstm(input_data, (h0, c0))
        Extract the last forward and backward outputs
        last_forward_output = output[:, -1, :hidden_size]
        last_backward_output = output[:, 0, hidden_size:]

        """
        h_n = h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size) #shape h_n into correct number
        if self.bidirectional:
            h = torch.cat((h_n[-1, -2, :, :], h_n[-1, -1, :, :]), dim=1) #take the forward and backward
        else:
            h = h_n[-1, -1, :, :]
        mean = self.fc_mean(h)#compress 256 numbers into 2
        log_var = self.fc_log_var(h)#compress 256 numbers into 2
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        return z, mean, log_var


class Decoder(nn.Module):
    def __init__(self,
                 input_size, #how many sensors
                 hidden_size,
                 latent_dim, #size of latnet vecotr
                 window_size, #how much time to reconrrust
                 dropout_lstm,
                 dropout_layer,
                 num_layers=1,
                 bidirectional=True
                 ):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.window_size = window_size
        self.p_lstm = dropout_lstm
        self.p_dropout_layer = dropout_layer
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm_to_hidden = nn.LSTM( #one layer constructed
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.p_lstm,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.dropout_layer = nn.Dropout(self.p_dropout_layer) #drop layers

        self.lstm_to_output = nn.LSTM( #get more accurate reconstruction again
            input_size=self.num_directions * hidden_size,
            hidden_size=input_size,
            batch_first=True
        )
    def forward(self, z): #in order to recronstrut the latent space
        latent_z = z.unsqueeze(1).repeat(1, self.window_size, 1)
        out, _ = self.lstm_to_hidden(latent_z) #expand the latent vector to the window size and run it through a LSTM to get back to the original input size
        out = self.dropout_layer(out) #
        out, _ = self.lstm_to_output(out) #compresses the output of the LSTM back to the original input size
        return out


class TSHAE(nn.Module): #

    def __init__(self, encoder, decoder=None, reconstruct=False): #decoder is optional set to false
        super(TSHAE, self).__init__()
        self.decode_mode = reconstruct
        if self.decode_mode:
            assert isinstance(decoder, nn.Module), "You should to pass a valid decoder" #decoder is optional
    
            self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x):#
        z, mean, log_var = self.encoder(x) #sensor data into latent vector
       
        if self.decode_mode:
            x_hat = self.decoder(z) #if decodre is true  reconstrust sensors
            return z, mean, log_var, x_hat

        return z, mean, log_var #latent vector, mean and log variance of the latent vector





    