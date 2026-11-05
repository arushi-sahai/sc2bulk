"""The data files' (Additional files 3-6) rules on synthetic frames (Genome Biology's 20 MB): the donor-level rule, the panels allowed to carry a sensitive column,
byte-reproducible zips. The files built from a whole tree are checked in tests/test_figures_end_to_end.py::test_s_data."""
import sys
import zipfile

import pandas as pd
import pytest

from sc2bulk.config import settings
from sc2bulk.config import figures as F

sys.path.insert(0, str(settings.REPO / "figures"))
import common  # noqa: E402
import data_files  # noqa: E402


def test_sensitive_columns():
    frame = pd.DataFrame({"sample_id": ["a"], "group": [1], "age_at_death": [70], "sex": ["f"]})
    assert data_files.sensitive(frame) == ["age_at_death", "sex"]
    assert data_files.sensitive(frame, F.PANEL_SENSITIVE["Fig. S7H"]) == []
    assert data_files.sensitive(frame, F.PANEL_SENSITIVE["Fig. S4I"]) == ["age_at_death", "sex"]
    assert common.citation("Fig. S7", "H") == "Fig. S7H" and common.citation("Fig 2", "A") == "Fig 2A"
    assert common.panel_name("Fig. S7", "H") == "S7HFig" and common.panel_name("Fig 2", "A") == "Fig2A"   # file names unchanged


def test_zip_is_reproducible_and_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LABELLED_OUTPUTS", tmp_path)
    files = {"x.csv": pd.DataFrame({"a": [1, 2]}), "y.csv": pd.DataFrame({"b": ["u"]})}
    descriptions = {"x.csv": ("Fig. S1a", "x"), "y.csv": ("Fig. S2a", "y")}
    data_files.write_zip("Additional file 9", files, descriptions)
    first = (tmp_path / "figures" / "data" / "Additional_file_9.zip").read_bytes()
    data_files.write_zip("Additional file 9", files, descriptions)
    assert (tmp_path / "figures" / "data" / "Additional_file_9.zip").read_bytes() == first
    with zipfile.ZipFile(tmp_path / "figures" / "data" / "Additional_file_9.zip") as archive:
        assert archive.namelist() == ["README.csv", "x.csv", "y.csv"]
    monkeypatch.setattr(data_files.S, "MAX_ADDITIONAL_FILE_BYTES", 10)
    with pytest.raises(AssertionError, match="20 MB"):
        data_files.write_zip("Additional file 9", files, descriptions)
