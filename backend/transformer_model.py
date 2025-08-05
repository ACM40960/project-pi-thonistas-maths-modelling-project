import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical

class StrokeMDN(nn.Module):
    def __init__(self, dim_in, dim_out, n_components):
        super().__init__()
        self.pi = nn.Linear(dim_in, n_components)
        self.mu = nn.Linear(dim_in, n_components * dim_out)
        self.sigma = nn.Linear(dim_in, n_components * dim_out)
        self.n_components = n_components
        self.dim_out = dim_out

    def forward(self, x):
        pi_logits = self.pi(x)  # (B, T, M)
        pi = F.log_softmax(pi_logits, dim=-1)

        mu = self.mu(x).view(*x.shape[:2], self.n_components, self.dim_out)
        sigma = torch.exp(self.sigma(x)).view(*x.shape[:2], self.n_components, self.dim_out)

        return pi, mu, sigma

class TransformerSketch(nn.Module):
    def __init__(self, embd=256, heads=4, layers=4, block_size=128, n_components=20):
        super().__init__()
        self.embd = embd
        self.stroke_proj = nn.Linear(2, embd)
        self.pen_emb = nn.Embedding(3, embd)
        self.pos_emb = nn.Embedding(block_size, embd)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embd, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.mdn = StrokeMDN(embd, 2, n_components)
        self.pen_head = nn.Linear(embd, 3)

    def forward(self, x):
        B, T, _ = x.size()
        pos = self.pos_emb(torch.arange(T, device=x.device))
        stroke = self.stroke_proj(x[:, :, :2])
        pen = self.pen_emb(x[:, :, 2].long())
        h = stroke + pen + pos
        h = self.transformer(h)
        pi, mu, sigma = self.mdn(h)
        pen_logits = self.pen_head(h)
        return pi, mu, sigma, pen_logits

    def compute_loss(self, x, target, mask):
        pi, mu, sigma, pen_logits = self.forward(x)

        # Stroke loss
        target_stroke = target[:, :, :2].unsqueeze(2)
        dist = torch.distributions.Normal(mu, sigma)
        log_prob = dist.log_prob(target_stroke).sum(-1)
        mdn_loss = -torch.logsumexp(pi + log_prob, dim=-1)

        # Pen loss
        pen_loss = F.cross_entropy(pen_logits.view(-1, 3), target[:, :, 2].long().view(-1), reduction='none')
        pen_loss = pen_loss.view(x.size(0), -1)

        total_loss = ((mdn_loss + pen_loss) * mask).sum() / mask.sum()
        return total_loss
