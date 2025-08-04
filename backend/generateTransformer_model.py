import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, OneHotCategorical, Categorical

# Hyperparameters (should be synced with training script)
block_size = 200  # max sequence length
embd = 384
embd_ffn = 4 * embd
num_heads = 6
n_layers = 6
n_components = 20
num_classes = 10

# === MDN Components ===
class NormalNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, n_components, full_cov=True):
        super().__init__()
        self.n_components = n_components
        self.out_dim = out_dim
        self.full_cov = full_cov
        self.tril_indices = torch.tril_indices(row=out_dim, col=out_dim, offset=0)
        self.mean_net = nn.Linear(in_dim, out_dim * n_components)
        if full_cov:
            self.tril_net = nn.Linear(in_dim, int(out_dim * (out_dim + 1) / 2 * n_components))
        else:
            self.tril_net = nn.Linear(in_dim, out_dim * n_components)

    def forward(self, x, tau=1.):
        mean = self.mean_net(x).view(x.shape[0], x.shape[1], self.n_components, self.out_dim)
        if self.full_cov:
            tril_values = self.tril_net(x).view(x.shape[0], x.shape[1], self.n_components, -1)
            tril = torch.zeros(x.shape[0], x.shape[1], self.n_components, self.out_dim, self.out_dim).to(x.device)
            tril[:, :, :, self.tril_indices[0], self.tril_indices[1]] = tril_values
            tril.diagonal(dim1=-2, dim2=-1)[:] = tril.diagonal(dim1=-2, dim2=-1).exp()
        else:
            tril = self.tril_net(x).view(x.shape[0], x.shape[1], self.n_components, -1)
            tril = torch.diag_embed(tril.exp())
        tril *= tau
        return MultivariateNormal(mean, scale_tril=tril)

class OneHotCategoricalNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.network = nn.Linear(in_dim, out_dim)

    def forward(self, x, tau=1.):
        logits = self.network(x) / tau
        return OneHotCategorical(logits=logits)

class CategoricalNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.network = nn.Linear(in_dim, out_dim)

    def forward(self, x, tau=1.):
        logits = self.network(x) / tau
        return Categorical(logits=logits)

class MDN(nn.Module):
    def __init__(self, dim_in, dim_out, n_components, full_cov=True):
        super().__init__()
        self.pi_net = OneHotCategoricalNetwork(dim_in, n_components)
        self.normal_net = NormalNetwork(dim_in, dim_out, n_components, full_cov)

    def forward(self, x, tau=1.):
        return self.pi_net(x, tau), self.normal_net(x, tau)

# === Transformer Core ===
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(embd, head_size, bias=False)
        self.key = nn.Linear(embd, head_size, bias=False)
        self.value = nn.Linear(embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v

class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embd, embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd, embd_ffn),
            nn.ReLU(),
            nn.Linear(embd_ffn, embd),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa_heads = MultiHead(num_heads, embd // num_heads)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(embd)
        self.ln2 = nn.LayerNorm(embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# === Transformer Sketch Model ===
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stroke_embed = nn.Linear(2, embd)
        self.pen_embed = nn.Embedding(3, embd)
        self.pos_embed = nn.Embedding(block_size, embd)
        self.class_embed = nn.Embedding(num_classes, embd)

        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embd)

        self.mdn_head = MDN(embd, 2, n_components)
        self.pen_head = nn.Linear(embd, 3)

    def forward(self, x, class_idx, tau=1.):
        B, T, C = x.shape
        stroke_emb = self.stroke_embed(x[:, :, :2])
        pen_emb = self.pen_embed(x[:, :, 2].long())
        pos_emb = self.pos_embed(torch.arange(T, device=x.device))

        class_token = self.class_embed(class_idx).unsqueeze(1)  # (B, 1, embd)
        token_emb = stroke_emb + pen_emb + pos_emb
        x = torch.cat([class_token, token_emb], dim=1)
        x = self.blocks(x)
        x = self.ln_f(x)

        # Discard class token output for prediction
        pi_net, normal_net = self.mdn_head(x[:, 1:], tau)
        pen_logits = self.pen_head(x[:, 1:])
        pen_dist = Categorical(logits=pen_logits)
        return pi_net, normal_net, pen_dist
