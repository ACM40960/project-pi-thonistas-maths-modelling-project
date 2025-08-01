import torch
import torch.nn as nn
import torch.nn.functional as F


class GenerateRNNEncoder(nn.Module):
    def __init__(self, input_size=15, hidden_size=256, latent_dim=128):  # 5 stroke + 10 class one-hot
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc_mu = nn.Linear(2 * hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(2 * hidden_size, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n: [2, B, H]
        h_fwd, h_bwd = h_n[0], h_n[1]  # [B, H]
        h = torch.cat([h_fwd, h_bwd], dim=1)  # [B, 2H]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class GenerateRNNDecoder(nn.Module):
    def __init__(self, input_size=15, hidden_size=512, latent_dim=128, num_mixtures=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_mixtures = num_mixtures
        self.lstm = nn.LSTM(input_size + latent_dim, hidden_size, batch_first=True ,num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_mixtures * 6 + 3)  # (μx, μy, σx, σy, ρxy, π) * M + 3 pen states

    def forward(self, x, z):
        B, T, _ = x.size()
        z = z.unsqueeze(1).repeat(1, T, 1)  # [B, T, latent_dim]
        x_in = torch.cat([x, z], dim=2)  # [B, T, input + latent]
        out, _ = self.lstm(x_in)
        out = self.fc(out)
        return out


class GenerateRNN(nn.Module):
    def __init__(self, input_size=15, hidden_size=256, latent_dim=128, num_mixtures=20):
        super().__init__()
        self.encoder = GenerateRNNEncoder(input_size, hidden_size, latent_dim)
        self.decoder = GenerateRNNDecoder(input_size, hidden_size*2, latent_dim, num_mixtures)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        output = self.decoder(x, z)
        return output, mu, logvar
