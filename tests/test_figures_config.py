"""config/figures.py and config/style.py: the registry, widths and heights, the
blinded-panel list, the numbered key, the n/a cells, the plan's citation order (check 1), and both config modules
outside every stage's code hash (editing them never invalidates a stage), and no figure reading configurations.py's
stale "color" field (config/style.MODEL_COLORS is the figures' one source)."""
import re
import sys

from sc2bulk import codehash
from sc2bulk.config import configurations, settings
from sc2bulk.config import figures as F
from sc2bulk.config import style as S
from sc2bulk.config.pairs import PAIR_ORDER

sys.path.insert(0, str(settings.REPO / "figures"))
import citation_order  # noqa: E402
import make_all  # noqa: E402


def test_registry():
    for item, spec in F.ITEMS.items():
        assert spec["script"] in (*make_all.SCRIPTS, "data_files", None), item
        if spec["script"] in make_all.SCRIPTS:
            assert spec["width"] in S.WIDTH_MM and item in F.PANELS and F.HEIGHT_MM[item] <= S.MAX_HEIGHT_MM, item
    assert set(F.PANELS) == {i for i, spec in F.ITEMS.items() if spec["script"] in make_all.SCRIPTS}
    for item, panels in F.BLINDED_PANELS.items():
        assert set(panels) <= set(F.PANELS[item]), item
    assert {p for p, _ in F.NOT_APPLICABLE["Fig 6C"]} == {"hammond"}
    assert set(F.FIG3A["columns"]) <= set(F.FIG3_CONDITIONS) <= set(F.S13_CONDITIONS) and F.FIG6B_PAIR in PAIR_ORDER
    assert set(F.ALIGNMENT_TICKS) == set(F.S13_CONDITIONS) and set(F.FIG3_KEYS) <= set(F.S13_KEYS["cells"])
    assert F.LAYOUT_PENDING <= set(F.PANELS)
    assert set(F.PANEL_SENSITIVE) <= {f"{item}{p}" for item in F.QC_FIGURES.values() for p in F.PANELS[item]}


def test_the_citation_plan():
    assert citation_order.check(F.CITATIONS) == []
    numbers = citation_order.additional_files(F.CITATIONS)   # figures 1, tables 2, data files 3-6
    assert {i: n for i, n in numbers.items() if n > 2} == {f"Additional file {n}": n for n in range(3, 7)}
    assert {n for i, n in numbers.items() if i.startswith("Fig. S")} == {1} and {n for i, n in numbers.items() if i.startswith("Table S")} == {2}


def test_outside_every_stage_hash():
    pipeline = settings.REPO / "pipeline"
    for script in pipeline.glob("[0-9]*.py"):
        stage = script.stem
        for pair in (*PAIR_ORDER, *([None] if stage not in ("02_preprocess", "02b_all_available") else [])):   # 02/02b always run per pair
            found = {str(p.relative_to(settings.REPO)) for p in codehash.files(script, pair, codehash.extra(stage, pair))}
            assert "src/sc2bulk/config/figures.py" not in found and "src/sc2bulk/config/style.py" not in found, (stage, pair)


def test_no_figure_reads_configuration_colors():
    """Every ["color"] lookup in figures/ is of a style dict, never configurations' MODELS / CONFIGURATIONS entries."""
    allowed = {"OTHER_TYPE", "REFERENCE_STYLE", "WATERMARK", "PENDING_WATERMARK", "PSEUDOBULK_LINE", "WITHIN_BULK_LINE"}
    for path in sorted((settings.REPO / "figures").glob("*.py")):
        text = path.read_text()
        bases = re.findall(r"(\w+)(?:\[[^\]]*\])*\[[\"']color[\"']\]", text)
        assert set(bases) <= allowed, (path.name, set(bases) - allowed)
        assert not re.search(r"\.get\([\"']color[\"']", text), path.name


def test_display_order_is_the_only_order():
    """DISPLAY_ORDER is a permutation of configurations.CONFIGURATION_ORDER (a renamed or added configuration can't
    silently drop out of a figure), and no figure script orders by CONFIGURATION_ORDER itself."""
    assert sorted(F.DISPLAY_ORDER) == sorted(configurations.CONFIGURATION_ORDER) and len(set(F.DISPLAY_ORDER)) == len(F.DISPLAY_ORDER)
    for path in sorted((settings.REPO / "figures").glob("*.py")):
        assert "CONFIGURATION_ORDER" not in path.read_text(), path.name
