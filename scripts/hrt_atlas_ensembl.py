"""The pinned housekeeping list: HRT Atlas v1.0's human housekeeping transcripts
(Hounkpe et al. 2021, Nucleic Acids Res 49:D947) as human Ensembl gene ids. Each transcript id (ENST) is mapped to
its gene id (ENSG) with the pinned Ensembl 111 GTF; transcripts retired before release 111 are counted and dropped,
not recovered. Writes config/attribution/hrt_atlas_v1.tsv: a "#" header naming the source and the counts, then
`gene` (the HRT symbol, for display only; ';'-joined where several map to one gene id) and `ensembl` (the gene id,
version stripped), one row per gene id, sorted. Both inputs are asserted against their SHA-256 pins. Re-pin
HOUSEKEEPING_SHA256 in config/genes.py with the SHA-256 it prints.

  python scripts/hrt_atlas_ensembl.py Housekeeping_GenesHuman.csv <RAW>/ensembl/Homo_sapiens.GRCh38.111.gtf.gz
"""
import argparse
import gzip
import hashlib
import re
from pathlib import Path

import pandas as pd

from sc2bulk.config import genes as G
from sc2bulk.config import sources

URL = "https://housekeeping.unicamp.br/Housekeeping_GenesHuman.csv"
DOWNLOADED = "2026-10-10"
SOURCE_SHA256 = "282244a1928e1c974cc16ab00ed8f27c4a000aebee532eb52a7d3f4592a4544f"


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transcript_genes(gtf) -> dict:
    """Transcript id -> gene id (version stripped), from the GTF's transcript lines."""
    genes = {}
    with gzip.open(gtf, "rt") as f:
        for line in f:
            fields = line.split("\t")
            if len(fields) > 8 and fields[2] == "transcript":
                transcript = re.search(r'transcript_id "([^"]+)"', fields[8]).group(1)
                genes[transcript] = re.search(r'gene_id "([^"]+)"', fields[8]).group(1).split(".")[0]
    return genes


def convert(csv, gtf) -> tuple:
    """The (gene, ensembl) table and the counts: transcript rows, retired transcripts dropped, genes."""
    hrt = pd.read_csv(csv, sep=";", keep_default_na=False)
    hrt["ensembl"] = hrt["Ensembl"].map(transcript_genes(gtf))
    kept = hrt.dropna(subset=["ensembl"])
    table = kept.groupby("ensembl")["Gene.name"].agg(lambda s: ";".join(sorted(set(s)))).reset_index()
    table = table.rename(columns={"Gene.name": "gene"})[["gene", "ensembl"]]
    return table, {"transcripts": len(hrt), "retired": len(hrt) - len(kept), "genes": len(table)}


def write(csv, gtf, out):
    table, counts = convert(csv, gtf)
    header = [
        "# HRT Atlas v1.0 human housekeeping genes (Hounkpe et al. 2021, Nucleic Acids Res 49:D947; GF spec §4.2; frozen §79).",
        f"# Source: {URL}, downloaded {DOWNLOADED},",
        f"#   SHA-256 {sha256(csv)} ({counts['transcripts']} transcript rows, semicolon-separated).",
        f"# Converted by scripts/hrt_atlas_ensembl.py: transcript ids mapped to gene ids with {Path(gtf).name}",
        f"#   (SHA-256 {sha256(gtf)}); {counts['retired']} transcripts retired before release {sources.ENSEMBL_RELEASE} dropped;",
        f"#   one row per gene id, sorted ({counts['genes']} genes). gene: the HRT symbol, for display only; ensembl: the gene id.",
    ]
    Path(out).write_text("\n".join(header) + "\n" + table.to_csv(sep="\t", index=False))
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("gtf")
    parser.add_argument("--out", default=str(G.HOUSEKEEPING_FILE))
    args = parser.parse_args()
    assert sha256(args.csv) == SOURCE_SHA256, "the HRT CSV differs from its pin"
    assert Path(args.gtf).name == sources.HUMAN_GTF and sha256(args.gtf) == sources.DOWNLOADS[("ensembl", sources.HUMAN_GTF)][-1], \
        "the GTF differs from its pin"
    print(write(args.csv, args.gtf, args.out), "SHA-256", sha256(args.out))
