"""run_all's stage table: the pooled stages run once without --pair (they read every pair)."""
import runpy

from sc2bulk.config import settings

R = runpy.run_path(str(settings.REPO / "pipeline" / "run_all.py"), run_name="run_all")


def test_pooled_stages_are_in_the_table_and_take_no_pair():
    assert R["POOLED"] <= set(R["STAGE_NAMES"])
    for stage in R["POOLED"]:   # each pooled script runs every pair itself (none needs --pair)
        script = next(s for name, s, *_ in R["STAGES"] if name == stage)
        text = (settings.REPO / "pipeline" / script).read_text()
        assert "required=True" not in text, f"{script} requires an argument"
