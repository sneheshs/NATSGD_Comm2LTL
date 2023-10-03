import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os


class HiddenLSTMEncoderDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, device):
        super(HiddenLSTMEncoderDecoder, self).__init__()

        self.num_layers = 1

        self.device = device
        # Encoder
        self.encoder = nn.LSTM(input_dim, latent_dim, num_layers=self.num_layers, bidirectional=True, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(input_dim, latent_dim, num_layers=self.num_layers, bidirectional=True, batch_first=True)

        self.hidden_to_embed = nn.Linear(latent_dim, input_dim)

    def _unflatten(self, x):
        return x.view(self.batch_size, self.num_layers, latent_dim).transpose(0, 1).contiguous()

    def _flatten(self, h):
        return h.transpose(0, 1).contiguous().view(self.batch_size, -1)

    def _unflatten_hidden(self, x):
        x_split = torch.split(x, int(x.shape[1] / 2), dim=1)
        h = (self._unflatten(x_split[0]), self._unflatten(x_split[1]))
        return h

    def _init_hidden_state(self, encoder_hidden):
        return tuple([self._concat_directions(h) for h in encoder_hidden])

    def _concat_directions(self, hidden):
        return hidden

    def _step(self, input, hidden):
        print([input.size()])
        output, hidden = self.decoder(input, hidden)

        output = self.hidden_to_embed(output.squeeze())
        return output, hidden

    def forward(self, x):
        x = x.to(self.device)
        self.batch_size, seq_len, features = x.size()
        _, (hidden, cell) = self.encoder(x)

        # batch_size, 2 * hidden_dim
        z = torch.cat([self._flatten(hidden), self._flatten(cell)], 1)

        # initialize the hidden state of the decoder
        hidden = self._unflatten_hidden(z)
        hidden = self._init_hidden_state(hidden)

        outputs = torch.zeros((self.batch_size, seq_len, features)).to(self.device)

        input = x[:, -1:, :]
       
        for i in range(seq_len):
            output, hidden = self._step(input, hidden)
            outputs[:, i:i + 1, :] = output.unsqueeze(1)
            input = x[:, i:i + 1, :]

        return outputs


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

learning_rate = 0.01
num_epochs = 10000
latent_dim = 512  # List of hidden dimensions for each layer
input_dim = 51 ###17x3


autoencoder = HiddenLSTMEncoderDecoder(input_dim, latent_dim, device).to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

dataset = np.load('data/NatSGD_v1.0.npz', allow_pickle=True)
data = dataset['data']
cols = dataset['fields']

padded_ges = torch.nn.utils.rnn.pad_sequence(data[:,4], batch_first=True, padding_value=0.0)
ges_lengths = torch.from_numpy(data[:,5].astype(float)).tolist()

# Train the autoencoder
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = autoencoder(padded_ges)  # Pass sequence_lengths
    loss = criterion(outputs, padded_ges.to(device))/padded_ges.shape[0]
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
 

model_path =os.getcwd() +  '/data/lstmEncoder_22.pth'


torch.save(autoencoder, model_path)
torch.save(autoencoder.state_dict(), os.getcwd() +  '/data/lstmEncoder_22.checkpoint')
