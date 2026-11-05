"""Set Transformer's network: the repository's classification configuration (main_pointcloud.py:
ISAB x 2 -> Dropout -> PMA(1 seed) -> Dropout -> Linear), assembled from the vendored, unedited
modules.py (third_party/set_transformer; CHANGES.txt there). One output logit; the embedding is
the PMA output (in eval mode), before the decoder SABs (decoder_sab: models.SetTransformer's two SABs
after its PMA, inside main_pointcloud's Dropouts), the last Dropout and Linear, as CytoSet's
embedding precedes its decoder. Search options:
encoder_block "isab" or "sab" (the paper's SAB encoder), encoder_layers, decoder_sab."""
import torch.nn as nn

from set_transformer.modules import ISAB, PMA, SAB


class Network(nn.Module):
    def __init__(self, config: dict, in_dim: int):
        super().__init__()
        d, h, m, ln = config["dim_hidden"], config["num_heads"], config["num_inds"], config["ln"]
        block = (lambda i: ISAB(i, d, h, m, ln=ln)) if config["encoder_block"] == "isab" else (lambda i: SAB(i, d, h, ln=ln))
        self.enc = nn.Sequential(*[block(in_dim if k == 0 else d) for k in range(config["encoder_layers"])])
        self.drop_in, self.pma = nn.Dropout(config["dropout"]), PMA(d, h, 1, ln=ln)
        self.dec = nn.Sequential(SAB(d, d, h, ln=ln), SAB(d, d, h, ln=ln)) if config["decoder_sab"] else nn.Identity()
        self.drop_out, self.linear = nn.Dropout(config["dropout"]), nn.Linear(d, 1)

    def embed(self, x, weights):
        return self.pma(self.drop_in(self.enc(x))).squeeze(1)

    def head(self, e):
        return self.linear(self.drop_out(self.dec(e[:, None, :]).squeeze(1))).squeeze(-1)


def network(config: dict, in_dim: int) -> Network:
    return Network(config, in_dim)
