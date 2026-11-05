"""The vendored CytoSet is the original, unchanged: its files match the upstream checksums,
and its pretrained model reproduces the result its README reports (HVTN test set:
accuracy 0.958, AUC 0.962)."""
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from cytoset.model import CytoSetModel

CYTOSET = Path(__file__).resolve().parents[1] / "third_party" / "cytoset"


def test_files_match_upstream():
    for line in (CYTOSET / "UPSTREAM_SHA256.txt").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((CYTOSET / name).read_bytes()).hexdigest() == digest, name


def test_pretrained_model_reproduces_published_result():
    model = CytoSetModel.from_pretrained(str(CYTOSET / "checkpoints" / "HVTN_model.pt"),
                                         str(CYTOSET / "config" / "model" / "ICS" / "config.json")).eval()
    data = pickle.loads((CYTOSET / "checkpoints" / "test_sample.pkl").read_bytes())
    with torch.no_grad():
        prob = np.array([float(model(torch.from_numpy(x))) for x in data["test_sample"]])
    label = np.array(data["test_phenotype"])
    assert round(float(((prob >= 0.5) == label).mean()), 3) == 0.958
    assert round(float(roc_auc_score(label, prob)), 3) == 0.962
