"""The scVI adapter (sc2bulk.encoders.scvi), in the scvi environment only: it trains with
validation on whole held-out samples, embeds samples, and thins a deep profile to the training
cells' median depth before embedding it."""
import numpy as np
import pytest

scvi = pytest.importorskip("scvi")

from sc2bulk.encoders import scvi as adapter


def counts(n, rng, depth=200):
    return rng.poisson(depth / 30, (n, 30))


def test_fit_and_embed():
    rng = np.random.default_rng(0)
    train, val = counts(300, rng), counts(100, rng)
    model, record = adapter.fit(train, val, {"n_latent": 4, "n_layers": 1}, seed=1, accelerator="cpu")
    assert record["epochs"] >= 1 and np.isfinite(record["validation_elbo"])
    samples = np.repeat(["a", "b", "c"], 100)
    z = adapter.embed_samples(model, train, samples, ["c", "a"])
    assert z.shape == (2, 4)
    deep = counts(2, rng, depth=200_000)
    thinned = adapter.embed_profiles(model, deep, record["median_library_size"], seed=2)
    assert thinned.shape == (2, 4) and np.isfinite(thinned).all()
    assert adapter.thin(deep[0], 200.0, np.random.default_rng(0)).sum() < 400
