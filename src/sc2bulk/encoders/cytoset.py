"""CytoSet's network (third_party/cytoset, unedited; changes listed in its CHANGES.txt).

The base and every pooling the original supports (mean, max, sum, at both sites) use the vendored
CytoSetModel's own layers. Added in this study: gated attention pooling (Ilse et al. 2018, attention
dimension h/2) at either site, and abundance weighting (a depth-weighted mean as the output
pooling). The inner-attention block reimplements PermEqui2's forward (Gamma x - Lambda pool(x)) with
attention as the pool; its attention scores are computed on Gamma x (width h, attention dimension
h/2, as at the output site), not on the 5,000-gene input. Logits are the decoder's output without the original's final sigmoid.
"""
import torch
import torch.nn as nn

from cytoset.model import Config, CytoSetModel, PermEqui2


class GatedAttention(nn.Module):
    """Attention weights over a bag's cells: softmax(w . (tanh(V h) * sigmoid(U h)))."""

    def __init__(self, dim):
        super().__init__()
        self.V, self.U, self.w = nn.Linear(dim, dim // 2), nn.Linear(dim, dim // 2), nn.Linear(dim // 2, 1)

    def forward(self, h):
        return torch.softmax(self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))), dim=1)


class AttentionBlock(nn.Module):
    """PermEqui2 with attention pooling: Gamma x - Lambda (attention-weighted sum of x), the weights
    scored on Gamma x."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Gamma, self.Lambda, self.attention = nn.Linear(in_dim, out_dim), nn.Linear(in_dim, out_dim, bias=False), GatedAttention(out_dim)

    def forward(self, x):
        g = self.Gamma(x)
        return g - self.Lambda((self.attention(g) * x).sum(dim=1, keepdim=True))


class Network(nn.Module):
    def __init__(self, config: dict, in_dim: int):
        super().__init__()
        args = Config.from_dict({"in_dim": in_dim, "h_dim": config["h_dim"], "nblock": config["nblock"],
                                 "pool": "mean" if config["pool"] == "attention" else config["pool"],
                                 "out_pool": "mean" if config["out_pool"] == "attention" else config["out_pool"]})
        base = CytoSetModel(args)
        if config["pool"] == "attention":
            layers, dim = [], in_dim
            for _ in range(config["nblock"]):
                layers += [AttentionBlock(dim, config["h_dim"]), nn.ELU(inplace=True)]
                dim = config["h_dim"]
            base.enc = nn.Sequential(*layers)
        self.enc, self.dec, self.out_pool = base.enc, base.dec, base.out_pool
        if config["decoder"] == "linear":   # the search's decoder_linear candidate: one Linear layer to the logit
            self.dec = nn.Linear(config["h_dim"], 1)
        self.output = "weighted" if config.get("abundance_weighting") else config["out_pool"]
        self.out_attention = GatedAttention(config["h_dim"]) if self.output == "attention" else None
        assert all(isinstance(m, (PermEqui2, AttentionBlock, nn.ELU)) for m in self.enc)

    def embed(self, x, weights):
        h = self.enc(x)
        if self.output == "weighted":
            return (weights.unsqueeze(-1) * h).sum(dim=1)
        if self.output == "attention":
            return (self.out_attention(h) * h).sum(dim=1)
        return self.out_pool(h, keepdim=False)

    def head(self, e):
        return self.dec(e).squeeze(-1)


def network(config: dict, in_dim: int) -> Network:
    return Network(config, in_dim)
