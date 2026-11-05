"""scripts/restamp_hashes.py on a synthetic outputs tree: a record is restamped only when its code hash is the
old byte hash of the current files, its original is kept, and stale or unrelated records are left alone."""
import json
import runpy

from sc2bulk import codehash, manifest
from sc2bulk.config.settings import REPO

R = runpy.run_path(str(REPO / "scripts" / "restamp_hashes.py"), run_name="restamp")


def write_record(path, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"settings": settings, "settings_hash": manifest.settings_hash(settings), "content_hash": "c", "size": 1}
    path.write_text(json.dumps(record, indent=1, sort_keys=True))
    return path


def test_restamp(tmp_path):
    old_04, new_04 = R["hashes"]("04_panel", "tms")
    old_11, new_11 = R["hashes"]("11_statistics", None)
    assert new_04 == codehash.stage_hash(REPO / "pipeline/04_panel.py", "tms") != old_04
    assert new_11 == codehash.stage_hash(REPO / "pipeline/11_statistics.py", None, codehash.extra("11_statistics", None))
    current = [write_record(tmp_path / "04_panel/tms/panel.csv.manifest.json", {"code": old_04, "n": 3}),
               write_record(tmp_path / "smoke/blinded_1/11_statistics/table.csv.manifest.json", {"code": old_11})]
    stale = write_record(tmp_path / "04_panel/hammond/panel.csv.manifest.json", {"code": "0123456789abcdef"})
    others = [write_record(tmp_path / "04_panel/tms/other.csv.manifest.json", {"n": 3}),            # no code hash
              write_record(tmp_path / "figures/fig2/a.csv.manifest.json", {"code": old_04}),        # no stage script
              write_record(tmp_path / "04_panel/tms/.tmp.x.csv.manifest.json", {"code": old_04})]   # a crashed write
    unreproducible = tmp_path / "04_panel/tms/odd.csv.manifest.json"
    unreproducible.write_text(json.dumps({"settings": {"code": old_04}, "settings_hash": "f" * 16}))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}

    out = R["restamp"](tmp_path, write=False)   # the dry run counts and writes nothing
    assert [p for p, _ in out["restamped"]] == sorted(current) and [p for p, _ in out["stale"]] == [stale]
    assert len(out["skipped"]) == 4
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before

    out = R["restamp"](tmp_path, write=True)
    assert [p for p, _ in out["restamped"]] == sorted(current)
    for path, new in zip(current, (new_04, new_11)):
        record = json.loads(path.read_text())
        assert record["settings"]["code"] == new and record["restamp"] == {"code": json.loads(before[path])["settings"]["code"]}
        assert record["settings_hash"] == manifest.settings_hash(record["settings"])
        assert path.with_name(path.name + ".pre_restamp").read_bytes() == before[path]
    for path in [stale, unreproducible, *others]:
        assert path.read_bytes() == before[path] and not path.with_name(path.name + ".pre_restamp").exists()

    out = R["restamp"](tmp_path, write=True)   # a second run finds them restamped and writes nothing
    assert not out["restamped"] and sum(why == "already restamped" for _, why in out["skipped"]) == 2


def test_old_digest_is_the_byte_hash(monkeypatch):
    files = codehash.files(REPO / "pipeline/04_panel.py", "tms")
    old = R["old_digest"](files)
    monkeypatch.setattr(codehash, "code", lambda path: path.read_bytes())
    assert codehash.digest(files) == old
