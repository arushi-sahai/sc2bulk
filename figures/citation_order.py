"""The citation-order check.

Genome Biology numbers figures and additional files by first citation in the main body (not the legends), and every
additional file must be cited there. The checks, on a sequence of (section, context, item) citations:
  1. per type (main Fig, Table, Fig. S, Table S, data file), the first main-body citations come in increasing number;
  2. every registered item is cited in the main body, and no unknown item is
     (a PLOS-style label left in the text, "S1 Fig", is an unknown item);
  3. config/figures.FIRST_CITED_IN: Table S12 first cited in the Methods, never in the Background;
  4. the Results cite Tables S3-S6 in order (RESULTS_ORDER);
  5. the legends' own order rules (LEGEND_ORDER: Fig 2's legend cites Figs. S9 -> S10 -> S11 -> S12);
  6. first citation numbers the Additional files as config/figures.ADDITIONAL_FILE does ( supplementary
     figures 1, tables 2; each data file its key's number).
Check 1 runs on config/figures.CITATIONS (tests/test_figures_config.py); check 2 on a manuscript:

  python figures/citation_order.py --manuscript PATH [PATH ...]   (Markdown or plain text, read in order)

which also compares the manuscript's sequence with the plan (a planned citation missing from the text is a warning:
the text is the authority) and checks "Additional file N" numbers against the mapping (additional_files()). The exit
code is nonzero on any failure."""
import argparse
import re
import sys
from pathlib import Path

from sc2bulk.config import figures as F

LEGENDS = "Legends"
ADDITIONAL_LIST = "Additional Files"   # the manuscript's list of additional files: not main body
TYPES = {"main": r"^Fig (\d+)$", "table": r"^Table (\d+)$", "s_fig": r"^Fig\. S(\d+)$", "s_table": r"^Table S(\d+)$",
         "s_data": r"^Additional file (\d+)$"}
SUPPLEMENTARY = ("s_fig", "s_table", "s_data")
# PLOS-style labels ("S1 Fig") are still parsed, so one left in the text fails as an unknown item
CITATION = re.compile(
    r"(?P<supp>\bS(?P<s1>\d+)[A-Za-z]?(?:\s*[–-]\s*S?(?P<s2>\d+)[A-Za-z]?)?\s+(?P<stype>Fig|Table|Data)\b)"
    r"|(?P<additional>\bAdditional file\s+(?P<a>\d+)(?::\s*)?)"
    r"|(?P<springer>\b(?P<sptype>Figs?\.?|Figures?|Tables?)\s+S(?P<p1>\d+)[A-Za-z]?(?:\s*[–-]\s*S?(?P<p2>\d+)[A-Za-z]?)?)"
    r"|(?P<main>\b(?P<mtype>Figs?\.?|Figures?|Tables?)\s+(?P<m1>\d+)[A-Za-z]?(?:\s*[–-]\s*(?P<m2>\d+)?[A-Za-z]?)?"
    r"(?P<more>(?:\s*(?:,|and)\s*\d+[A-Za-z]?\b)*))")
LEGEND_LABEL = re.compile(r"^\W*(Fig(?:ure)?\.?\s+S?\d+|Table\s+S?\d+)\.")


def kind(item):
    for name, pattern in TYPES.items():
        match = re.match(pattern, item)
        if match:
            return name, int(match.group(1))
    return None, None


def main_body(sequence) -> list:
    return [c for c in sequence if c[0] in F.SECTIONS]


def first_citations(sequence) -> list:
    """Items in order of first main-body citation."""
    seen = []
    for _, _, item in main_body(sequence):
        if item not in seen:
            seen.append(item)
    return seen


def check(sequence) -> list:
    """Every rule's failures (strings), in rule order."""
    failures = []
    first = first_citations(sequence)
    unknown = sorted({item for _, _, item in sequence} - set(F.ITEMS))
    failures += [f"unknown item: {item}" for item in unknown]
    for name in TYPES:
        numbers = [kind(i)[1] for i in first if kind(i)[0] == name]
        if numbers != sorted(numbers):
            failures.append(f"{name}: first cited out of order: {numbers}")
    failures += [f"never cited in the main body: {item}" for item in F.ITEMS if item not in first]
    for item, section in F.FIRST_CITED_IN.items():
        sections = [s for s, _, i in main_body(sequence) if i == item]
        if sections and sections[0] != section:
            failures.append(f"{item}: first cited in {sections[0]}, not {section}")
        if "Background" in sections:
            failures.append(f"{item}: cited in the Background")
    results = [i for i in first_citations([c for c in sequence if c[0] == "Results"]) if i in F.RESULTS_ORDER]
    if results != [i for i in F.RESULTS_ORDER if i in results]:
        failures.append(f"Results cite {F.RESULTS_ORDER} out of order: {results}")
    for legend, order in F.LEGEND_ORDER.items():
        cited = first_citations([("Results", "", i) for s, c, i in sequence if s == LEGENDS and c == legend and i in order])
        if cited != [i for i in order if i in cited]:
            failures.append(f"{legend}'s legend cites {order} out of order: {cited}")
    for item, number in additional_files(sequence).items():
        if number != scheme(item):
            failures.append(f"{item}: first citation makes it Additional file {number}, frozen §72 Additional file {scheme(item)}")
    return failures


def scheme(item) -> int:
    """The item's Additional file: 1 for a supplementary figure, 2 for a supplementary table, a data
    file its own number."""
    name, number = kind(item)
    return F.ADDITIONAL_FILE.get(name, number)


def additional_files(sequence) -> dict:
    """item -> Additional file number by first main-body citation: every supplementary figure shares one file, every
    supplementary table another (config/figures.ADDITIONAL_FILE); each data file is its own."""
    numbers, out = {}, {}
    for item in first_citations(sequence):
        if kind(item)[0] not in SUPPLEMENTARY or item not in F.ITEMS:
            continue
        unit = kind(item)[0] if kind(item)[0] in F.ADDITIONAL_FILE else item
        numbers.setdefault(unit, len(numbers) + 1)
        out[item] = numbers[unit]
    return out


def expand(match) -> list:
    """The items one regex match names (ranges and lists expanded)."""
    if match["supp"]:
        last = int(match["s2"] or match["s1"])
        return [f"S{n} {match['stype']}" for n in range(int(match["s1"]), last + 1)]
    if match["springer"]:
        stype = "Table" if match["sptype"].startswith("Table") else "Fig."
        return [f"{stype} S{n}" for n in range(int(match["p1"]), int(match["p2"] or match["p1"]) + 1)]
    mtype = "Table" if match["mtype"].startswith("Table") else "Fig"
    numbers = list(range(int(match["m1"]), int(match["m2"] or match["m1"]) + 1))
    numbers += [int(n) for n in re.findall(r"(\d+)[A-Za-z]?", match["more"] or "")]
    return [f"{mtype} {n}" for n in numbers]


def parse(text) -> tuple:
    """([(section, context, item)] in reading order, [(Additional file number, the item it names or None)]). Sections
    come from headings (Markdown "#" or a line that is just the section's name: a main section, "Additional files" or a
    heading with "legend"); other headings are subsections. In the legends each paragraph's own label ("Fig 2.") sets
    the context and isn't a citation.
    "Additional file N" followed by an item ("Additional file 1: Fig. S8a") names that item; on its own it cites a
    data file ("Additional file 3"), or names nothing if no item has that key (Additional files 1 and 2)."""
    sequence, additional = [], []
    section, context = None, ""
    for line in text.splitlines():
        heading = re.match(r"^\s*#*\s*([A-Za-z ]+?)\s*#*\s*$", line)
        if heading and ("legend" in heading.group(1).lower() or heading.group(1).strip().title() in (*F.SECTIONS, ADDITIONAL_LIST)):
            section = LEGENDS if "legend" in heading.group(1).lower() else heading.group(1).strip().title()
            context = ""
            continue
        if section == LEGENDS:
            label = LEGEND_LABEL.match(line)
            if label:
                context = expand(CITATION.search(label.group(1)))[0]
                line = line[label.end():]
        matches = list(CITATION.finditer(line))
        for i, match in enumerate(matches):
            if match["additional"]:
                follows = matches[i + 1] if i + 1 < len(matches) and matches[i + 1].start() == match.end() else None
                own = f"Additional file {match['a']}" if follows is None and f"Additional file {match['a']}" in F.ITEMS else None
                additional.append((int(match["a"]), expand(follows)[0] if follows is not None else own))
                sequence += [(section, context, own)] if own else []
                continue
            sequence += [(section, context, item) for item in expand(match)]
    return sequence, additional


def check_manuscript(paths_) -> tuple:
    """(failures, warnings) for the manuscript files, read in order."""
    sequence, additional = parse("\n".join(Path(p).read_text() for p in paths_))
    failures = check(sequence)
    mapping = additional_files(sequence)
    numbers = []
    for number, item in additional:
        numbers += [number] if number not in numbers else []
        if item is not None and mapping.get(item) != number:
            failures.append(f"Additional file {number} names {item}, which is Additional file {mapping.get(item)}")
    if numbers != sorted(numbers):
        failures.append(f"Additional files first cited out of order: {numbers}")
    cited = {(s, i) for s, _, i in sequence}
    warnings = [f"planned citation missing: {s} {c} -> {i}" for s, c, i in F.CITATIONS if (s, i) not in cited]
    return failures, warnings


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", nargs="+", required=True)
    failures, warnings = check_manuscript(parser.parse_args(argv).manuscript)
    for line in warnings:
        print(f"warning: {line}")
    for line in failures:
        print(f"FAIL: {line}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
