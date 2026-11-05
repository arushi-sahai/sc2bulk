"""A work unit counts as done only when its output exists, with its recorded size, and was made
with the same settings; the manifest records the output's content hash, the environment and the
unit's cost."""
import json

import pytest

from sc2bulk import manifest


def test_work_unit_resumes_only_on_matching_settings(tmp_path):
    output = tmp_path / "unit.csv"
    settings = {"pair": "tms", "seed": 12340}
    assert not manifest.is_done([output], settings)

    manifest.temporary(output).write_text("x\n")
    assert not manifest.is_done([output], settings)   # nothing is done until finish()
    unit = manifest.begin()
    manifest.finish(output, settings, unit, seed=12340)

    assert output.read_text() == "x\n"
    assert not manifest.temporary(output).exists()
    assert manifest.is_done([output], settings)
    cost = manifest.read(output)["cost"]
    assert cost["device"] == "cpu" and cost["gpu"] is None and cost["seconds"] >= 0 and cost["peak_rss_kb"] > 0
    assert not manifest.is_done([output], {**settings, "seed": 12341})
    assert not manifest.is_done([output, tmp_path / "other.csv"], settings)   # every output must exist

    record = manifest.read(output)
    assert record["content_hash"] == manifest.content_hash(output) and record["size"] == 2
    assert record["blinded"] is None and record["environment"]["fingerprint"]
    output.write_text("x\nmore\n")                    # a truncated or altered copy is not done
    assert not manifest.is_done([output], settings)


def test_input_hashes_are_contents(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest.config, "OUTPUTS", tmp_path)
    output = tmp_path / "stage" / "out.csv"
    output.parent.mkdir()
    manifest.temporary(output).write_text("y\n")
    manifest.finish(output, {}, manifest.begin())
    assert manifest.input_hashes([output]) == {"stage/out.csv": manifest.content_hash(output)}


def test_label_readers_write_only_label_bearing_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest.config, "LABELLED_OUTPUTS", tmp_path)
    monkeypatch.setattr(manifest, "LABELS_READ", True)
    plain = tmp_path / "scores.csv"
    manifest.temporary(plain).write_text("z\n")
    with pytest.raises(AssertionError):
        manifest.finish(plain, {}, manifest.begin())
    labelled = tmp_path / f"{manifest.config.LABELLED}_scores.csv"
    manifest.temporary(labelled).write_text("z\n")
    manifest.finish(labelled, {}, manifest.begin())


def test_check_environments(tmp_path):
    for name, fingerprint in (("a", "1"), ("b", "2")):
        (tmp_path / f"{name}.csv.manifest.json").write_text(json.dumps({"environment": {"name": "sc2bulk", "fingerprint": fingerprint}}))
    with pytest.raises(AssertionError):
        manifest.check_environments(tmp_path)
    (tmp_path / "b.csv.manifest.json").write_text(json.dumps({"environment": {"name": "fm-scgpt", "fingerprint": "2"}}))
    manifest.check_environments(tmp_path)
    (tmp_path / "smoke").mkdir()                      # the smoke run's outputs don't count
    (tmp_path / "smoke" / "c.csv.manifest.json").write_text(json.dumps({"environment": {}}))
    manifest.check_environments(tmp_path)
