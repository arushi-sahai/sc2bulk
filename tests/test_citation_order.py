"""The citation-order check on synthetic manuscripts:
main-body numbering, completeness, Table S12 in the Methods, Tables S3-S6 in order, Fig 2's legend order, ranges and
Genome Biology forms, and the Additional-file numbering."""
import sys

from sc2bulk.config import settings
from sc2bulk.config import figures as F

sys.path.insert(0, str(settings.REPO / "figures"))
import citation_order  # noqa: E402

LEGEND = ("# Figure legends\n**Fig 2.** Title. Read-outs: Fig. S9; searches: Fig. S10; robustness: Fig. S11; "
          "full genes: Fig. S12a.\n")


def manuscript(order=None) -> str:
    """The plan written out as a manuscript, sections as headings, one citation per line."""
    lines, section = [], None
    for s, _, item in order or F.CITATIONS:
        if s != section:
            lines.append(f"# {s}")
            section = s
        lines.append(f"As shown ({item}).")
    return "\n".join(lines) + "\n" + LEGEND


def check(text, tmp_path):
    path = tmp_path / "m.md"
    path.write_text(text)
    return citation_order.check_manuscript([path])


def test_the_plan_passes(tmp_path):
    failures, warnings = check(manuscript(), tmp_path)
    assert failures == [] and warnings == []


def test_failures(tmp_path):
    swapped = list(F.CITATIONS)
    i, j = swapped.index(("Results", "Fig 1D", "Fig. S2")), swapped.index(("Results", "Fig 1D", "Fig. S3"))
    swapped[i], swapped[j] = swapped[j], swapped[i]
    assert any("s_fig: first cited out of order" in f for f in check(manuscript(swapped), tmp_path)[0])
    background = [("Background", "", "Table S12"), *F.CITATIONS]
    failures = check(manuscript(background), tmp_path)[0]
    assert any("Table S12: first cited in Background" in f for f in failures) and any("cited in the Background" in f for f in failures)
    missing = [c for c in F.CITATIONS if c[2] != "Table S8"]
    assert "never cited in the main body: Table S8" in check(manuscript(missing), tmp_path)[0]
    assert "unknown item: Fig. S30" in check(manuscript() + "# Results\nSee Fig. S30.\n", tmp_path)[0]
    assert "unknown item: S9 Fig" in check(manuscript() + "# Results\nSee S9 Fig.\n", tmp_path)[0]   # a PLOS label left over
    tables = list(F.CITATIONS)
    i, j = tables.index(("Results", "Fig 2", "Table S3")), tables.index(("Results", "Fig 2", "Table S4"))
    tables[i], tables[j] = tables[j], tables[i]
    assert any("Results cite" in f for f in check(manuscript(tables), tmp_path)[0])
    legend = manuscript().replace("robustness: Fig. S11; full genes: Fig. S12a", "full genes: Fig. S12a; robustness: Fig. S11")
    assert any("Fig 2's legend" in f for f in check(legend, tmp_path)[0])
    only_legend = manuscript([c for c in F.CITATIONS if c[2] != "Fig. S9"])   # Fig. S9 cited in Fig 2's legend only
    assert "never cited in the main body: Fig. S9" in check(only_legend, tmp_path)[0]


def test_parsing_forms():
    sequence, additional = citation_order.parse(
        "# Results\nSee Figs. S2–S7, Figs 2A–C and Fig 2A and 3C, Tables S3-S6, Table 1, Fig. S12a, Table S1B.\n"
        "Also Fig. 4a, Figs. 5a–c, Figure 6, Additional file 1: Fig. S4, Additional file 2: Table S2, Additional file 3.\n")
    items = [i for _, _, i in sequence]
    assert items == ["Fig. S2", "Fig. S3", "Fig. S4", "Fig. S5", "Fig. S6", "Fig. S7", "Fig 2", "Fig 2", "Fig 3", "Table S3", "Table S4",
                     "Table S5", "Table S6", "Table 1", "Fig. S12", "Table S1", "Fig 4", "Fig 5", "Fig 6", "Fig. S4", "Table S2",
                     "Additional file 3"]
    assert additional == [(1, "Fig. S4"), (2, "Table S2"), (3, "Additional file 3")]
    sequence, additional = citation_order.parse("# Results\nSee Additional file 1.\n")
    assert sequence == [] and additional == [(1, None)]   # the figures' file as a whole names no item
    sequence, _ = citation_order.parse("# Figure legends\nFig 3. A title citing Fig. S13.\n")
    assert sequence == [("Legends", "Fig 3", "Fig. S13")]   # the legend's own label is context, not a citation
    sequence, _ = citation_order.parse("# Results\n## A subsection\nFig. S1.\n# Additional files\nAdditional file 1: Fig. S1.\n")
    assert [s for s, _, _ in sequence] == ["Results", "Additional Files"]   # subsections stay in Results; the list is not main body


def test_additional_files(tmp_path):
    sequence = [("Results", "", "Fig. S1"), ("Results", "", "Table S1"), ("Results", "", "Fig. S2"), ("Methods", "", "Additional file 3")]
    assert citation_order.additional_files(sequence) == {"Fig. S1": 1, "Table S1": 2, "Fig. S2": 1, "Additional file 3": 3}
    tables_first = [("Results", "", "Table S1"), *sequence]   # a table before any figure: the scheme's numbers break
    assert any("first citation makes it Additional file 2, frozen §72 Additional file 1" in f for f in citation_order.check(tables_first))
    text = manuscript().replace("As shown (Fig. S1).", "As shown (Additional file 2: Fig. S1).")
    assert any("Additional file 2 names Fig. S1" in f for f in check(text, tmp_path)[0])
    text = manuscript().replace("As shown (Fig. S1).", "As shown (Additional file 1: Fig. S1).")
    assert check(text, tmp_path)[0] == []
