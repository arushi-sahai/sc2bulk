"""ScRAT's network: the authors' TransformerPredictor (github.com/yuzhenmao/ScRAT at
training.SCRAT_COMMIT), imported from a clone the user fetches (SC2BULK_SCRAT; never copied into this
repository: the upstream has no licence file). Constructed exactly as its main.py does
(one output; input_dropout, its constructor option that main.py leaves at 0, is a search candidate). The embedding is the cells' mean after the encoder, the input of
its output_net (the site of its forward's x.mean(1)); the head is output_net. The encoder is
called with task='train' (its default; its other path's top-k indexing fails below 10 cells, which
includes a one-profile bulk bag)."""
import subprocess
import sys

import torch.nn as nn

from sc2bulk.config import training as T


def load(repo: str):
    """The authors' TransformerPredictor, from a clone at the pinned commit (asserted)."""
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head == T.SCRAT_COMMIT, f"ScRAT clone at {head}, not {T.SCRAT_COMMIT}"
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
    assert not dirty, "the ScRAT clone's tracked files have local changes"
    sys.path.insert(0, repo)
    from Transformer import TransformerPredictor   # the authors' module
    return TransformerPredictor


class Network(nn.Module):
    def __init__(self, config: dict, in_dim: int, repo: str):
        super().__init__()
        in_dim = T.SCRAT_PCA_COMPONENTS if config["pca"] else in_dim   # harness.Inputs projects the cells
        self.model = load(repo)(input_dim=in_dim, model_dim=config["model_dim"], num_classes=1,
                                num_heads=config["num_heads"], num_layers=config["num_layers"], dropout=config["dropout"],
                                input_dropout=config["input_dropout"], pca=config["pca"], norm_first=config["norm_first"])

    def embed(self, x, weights):
        m = self.model
        x = m.input_net(x) if m.hparams.pca else m.dimRedu_net(x)
        return m.transformer(x, mask=None, task="train").mean(1)

    def head(self, e):
        return self.model.output_net(e).squeeze(-1)


def network(config: dict, in_dim: int, repo: str) -> Network:
    return Network(config, in_dim, repo)
