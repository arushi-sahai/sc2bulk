"""Fig 1A, Fig 1B and Fig. S1A-D: the schematics, drawn as vector graphics in BioRender's flat style (rounded boxes,
soft fills, thin outlines, filled arrowheads; text only, no icons; 7-8 pt at print size). They replace the BioRender AI
drafts (whose graphics stayed one flat image). Label-free: no stage
output is read. Each function draws into axes whose data coordinates are millimetres at print size (SIZES; origin
bottom left); fig1.py and s01.py place them.

  python figures/schematics.py OUT_DIR       previews: each panel on its own as .pdf and .png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import common
import style
from sc2bulk.config import style as S

MM = 1 / 25.4
INK = "#333333"                        # outlines, arrows and text
GREY = "#D9D9D9"                       # neutral fills (Fig 1A is greys only)
GREY_DARK = "#9E9E9E"
LIGHT = "#F2F2F2"                      # discarded heads
BODY, SMALL = 8, 7                     # pt at print size
LINE = 0.6                             # pt
SEED = 3                               # the shading of the grey input matrices


def mm_axes(ax, width_mm, height_mm):
    """Data coordinates in millimetres at print size (origin bottom left), no frame."""
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    ax.axis("off")


def box(ax, x, y, w, h, fill=GREY, edge=INK, dashed=False, radius=1.2, lw=LINE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
                                edgecolor=edge, linewidth=lw, linestyle=(0, (3, 2)) if dashed else "-"))


def on(fill) -> str:
    """Text color on a fill: white on dark fills, ink on light ones (white on #F0A35E or #CC79A7 is hard to read)."""
    r, g, b = to_rgb(fill)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK


def text(ax, x, y, s, size=BODY, color=INK, ha="center", va="center", **kwargs):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, linespacing=1.15, **kwargs)


def arrow(ax, start, end, color=INK, dashed=False, lw=LINE):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>,head_length=2.2,head_width=1.1", mutation_scale=1,
                                 color=color, linewidth=lw, linestyle=(0, (3, 2)) if dashed else "-",
                                 shrinkA=0, shrinkB=0))


def matrix(ax, x, y, ncol, nrow, cell, fill, edge="white", shades=None):
    """A gene x cell grid; shades (nrow x ncol, 0-1) vary the fill's lightness, as BioRender's heat matrices do."""
    for i in range(nrow):
        for j in range(ncol):
            alpha = 1.0 if shades is None else 0.35 + 0.65 * shades[i, j]
            ax.add_patch(Rectangle((x + j * cell, y + i * cell), cell, cell, facecolor=fill, alpha=alpha,
                                   edgecolor="none"))
    for k in range(ncol + 1):
        ax.plot([x + k * cell] * 2, [y, y + nrow * cell], color=edge, linewidth=0.5)
    for k in range(nrow + 1):
        ax.plot([x, x + ncol * cell], [y + k * cell] * 2, color=edge, linewidth=0.5)
    ax.add_patch(Rectangle((x, y), ncol * cell, nrow * cell, facecolor="none", edgecolor=INK, linewidth=0.4))


def bar(ax, x, y, w, h, fill, edge=INK):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=0.4))


def dot(ax, x, y, filled, r=0.9):
    ax.add_patch(Circle((x, y), r, facecolor=INK if filled else "white", edgecolor=INK, linewidth=0.5))


def fig1a(ax):
    """The task: a sample's cells become one vector; a ridge readout trained on single-cell samples is applied unchanged
    to bulk samples, already one profile each, and scored by bulk AUROC (170 x 45 mm, greys only)."""
    mm_axes(ax, 170, 45)
    rng = np.random.default_rng(SEED)
    # training row: a stack of samples, each a cells x genes matrix
    text(ax, 1, 38.5, "Training:", ha="left", weight="bold")
    text(ax, 1, 35, "single-cell samples", ha="left")
    text(ax, 1, 26, "each sample is a\nmatrix of cells", size=SMALL, ha="left")
    for k in (2, 1):
        bar(ax, 29 + 1.4 * k, 26 + 1.4 * k, 14, 12, "white")
    matrix(ax, 29, 26, 7, 6, 2.0, GREY_DARK, shades=rng.random((6, 7)))
    text(ax, 27.3, 32, "genes", size=SMALL, rotation=90)
    text(ax, 36, 24.3, "cells", size=SMALL)
    dot(ax, 47, 40.3, False)
    dot(ax, 49.4, 40.3, True)
    arrow(ax, (49, 32), (61, 32))
    text(ax, 55, 34.3, "compress", size=SMALL)
    # testing row: bulk samples, already one profile each
    text(ax, 1, 16, "Testing:", ha="left", weight="bold")
    text(ax, 1, 12.5, "bulk samples", ha="left")
    text(ax, 1, 4.5, "already one profile\nper sample", size=SMALL, ha="left")
    for i, filled in enumerate((False, True, False, True)):
        bar(ax, 30.5 + 3.4 * i, 2.5, 1.6, 12, GREY_DARK)
        dot(ax, 31.3 + 3.4 * i, 17, filled, r=0.75)
    arrow(ax, (44, 9), (61, 9))   # from the bulk profiles to the representation
    # the shared representation
    box(ax, 61, 3, 30, 38, radius=2)
    text(ax, 76, 36, "Sample\nrepresentation")
    for i, shade in enumerate(("#3A3A3A", "#666666", "#8C8C8C", "#B3B3B3", "#E6E6E6")):
        bar(ax, 66.5 + 3.8 * i, 24.5, 3.8, 3.8, shade)
    text(ax, 76, 19.5, "one vector\nper sample", size=SMALL)
    text(ax, 76, 9, "pseudobulk, or\na learned model", size=SMALL)
    # the ridge readout: trained on single-cell samples, applied unchanged to bulk
    arrow(ax, (91, 22), (100, 22))
    box(ax, 100, 15, 18, 14)
    text(ax, 109, 22, "Ridge\nreadout")
    text(ax, 109, 43, "labels", size=SMALL)
    arrow(ax, (109, 41.3), (109, 29))
    text(ax, 107.5, 35, "trained on\nsingle-cell\nsamples", size=SMALL, ha="right")
    text(ax, 109, 9.5, "applied unchanged\nto bulk samples", size=SMALL)
    # the predicted sample label, scored by bulk AUROC
    arrow(ax, (118, 22), (124, 22))
    box(ax, 124, 13, 26, 18)
    text(ax, 137, 34, "Sample label")
    dot(ax, 127.5, 25.8, False)
    text(ax, 130, 25.8, "control / younger", size=SMALL, ha="left")
    dot(ax, 127.5, 18.2, True)
    text(ax, 130, 18.2, "case / older", size=SMALL, ha="left")
    arrow(ax, (150, 22), (155, 22))
    ax.plot([156, 156, 169], [28, 15, 15], color=INK, linewidth=LINE)
    ax.plot([156, 169], [15, 28], color=GREY_DARK, linewidth=0.5, linestyle=(0, (2, 1.5)))
    t = np.linspace(0, 1, 50)
    ax.plot(156 + 13 * t, 15 + 13 * (1 - (1 - t) ** 3), color=INK, linewidth=0.8)
    text(ax, 162.5, 34, "Bulk\nAUROC")


def fig1b(ax):
    """How each family turns a sample's cells into one vector (88 x 52 mm, family colors)."""
    mm_axes(ax, 88, 52)
    rows = [  # family, family label, model box (None: the sum), note under the box, output
        ("pseudobulk", "Pseudobulk", None, "sum over cells", "pseudobulk profile"),
        ("set_encoder", "Supervised set\nencoders", "CytoSet, Set\nTransformer, ScRAT", "trained on single-cell labels",
         "set embedding"),
        ("generative", "Unsupervised\ngenerative model", "scVI\n(per cell)", "fit without labels", "mean embedding"),
        ("foundation_model", "Frozen foundation\nmodels", "Geneformer, scGPT\n(per cell)", "pretrained",
         "mean embedding"),
    ]
    for r, (key, family, model, note, output) in enumerate(rows):
        color = S.FAMILY_COLORS[key]
        y = 44.5 - 12 * r              # row centre
        text(ax, 0.5, y, family, size=BODY, color=color, ha="left", weight="bold")
        matrix(ax, 27, y - 3.5, 4, 4, 1.75, color)
        if r == 0:
            text(ax, 30.5, y + 5, "cells", size=SMALL)
            text(ax, 25.6, y, "genes", size=SMALL, rotation=90)
        arrow(ax, (34.5, y), (38, y))
        if model is None:
            ax.add_patch(Circle((41.5, y), 3.3, facecolor=color, edgecolor=INK, linewidth=LINE))
            text(ax, 41.5, y, "Σ", size=10, color=on(color), weight="bold")
            text(ax, 41.5, y - 5.6, note, size=SMALL)
            arrow(ax, (44.8, y), (65.5, y))
        else:
            box(ax, 38, y - 4, 25, 8, fill=color)
            text(ax, 50.5, y, model, size=SMALL, color=on(color))
            text(ax, 50.5, y - 5.6, note, size=SMALL)
            arrow(ax, (63, y), (65.5, y))
        bar(ax, 66, y - 1.2, 21.5, 2.4, color)
        text(ax, 76.75, y - 4.2, output, size=SMALL)


def encoder_row(ax, steps, color, heading, input_label, gap=4.5, bar_w=12, readout_w=14, y=6, h=9):
    """S1A-C: input -> searched (dashed) or fixed steps -> embedding -> ridge readout, laid out left to right from x = 1
    with a fixed gap for each arrow; the heading top left and the discarded head above the embedding, so nothing hangs
    below the row (rows stack closely)."""
    top = y + h / 2 + 3                # the head box's bottom edge
    text(ax, 1, top + h / 2, heading, size=BODY, color=color, ha="left", weight="bold")
    matrix(ax, 4, y - 4, 5, 5, 1.6, GREY_DARK, shades=np.random.default_rng(SEED).random((5, 5)))
    text(ax, 8, y + 5.6, input_label, size=SMALL)
    x = 12
    for label, searched, width in steps:
        arrow(ax, (x, y), (x + gap, y))
        x += gap
        box(ax, x, y - h / 2, width, h, fill=color, dashed=searched)
        text(ax, x + width / 2, y, label, size=SMALL, color=on(color))
        x += width
    arrow(ax, (x, y), (x + gap, y))
    x += gap
    bar(ax, x, y - 1.2, bar_w, 2.4, color)
    text(ax, x + bar_w / 2, y - 3.6, "embedding", size=SMALL)
    arrow(ax, (x + bar_w / 2, y + 1.2), (x + bar_w / 2, top), color=GREY_DARK)
    box(ax, x + bar_w / 2 - 8, top, 16, h, fill=LIGHT, edge=GREY_DARK)   # solid: dashes mean "searched"
    text(ax, x + bar_w / 2, top + h / 2, "head\n(discarded)", size=SMALL, color=GREY_DARK)
    x += bar_w
    arrow(ax, (x, y), (x + gap, y))
    x += gap
    box(ax, x, y - h / 2, readout_w, h)
    text(ax, x + readout_w / 2, y, "ridge\nreadout", size=SMALL)
    return x + readout_w


def s1a(ax):
    """CytoSet at its base configuration; dashed: varied by the configuration search (170 x 24 mm)."""
    mm_axes(ax, 170, 24)
    encoder_row(ax, [("permutation-\nequivariant block", True, 26), ("max pooling\n(within block)", True, 22),
                           ("mean pooling\n(over cells)", True, 22)], S.MODEL_COLORS["cytoset"], "CytoSet", "bag of cells",
                      gap=7, bar_w=16, readout_w=16)
    box(ax, 140, 17, 29, 6, fill="white", dashed=True)   # the key, top right, read as a legend
    text(ax, 154.5, 20, "dashed = searched", size=SMALL)


def s1b(ax):
    """Set Transformer at its base configuration (85 x 24 mm)."""
    mm_axes(ax, 85, 24)
    encoder_row(ax, [("ISAB ×2", True, 13), ("PMA", False, 10)], S.MODEL_COLORS["set_transformer"], "Set Transformer",
                "bag of cells")


def s1c(ax):
    """ScRAT at its base configuration (85 x 24 mm)."""
    mm_axes(ax, 85, 24)
    encoder_row(ax, [("PCA", True, 7.5), ("transformer", True, 15), ("mean\npooling", False, 9.5)], S.MODEL_COLORS["scrat"],
                "ScRAT", "bag of cells", gap=3.2, bar_w=9, readout_w=12)


def tokens(ax, x, y, labels, color, values=None, w=5.6, h=3.6):
    """A token sequence (one box per token, in the model's color; a second row of value tokens below if given)."""
    for k, label in enumerate(labels):
        left = x + k * (w + 0.8)
        top = y if values is None else y + h / 2 + 0.3
        box(ax, left, top - h / 2, w, h, fill="white", edge=color, radius=0.6)
        text(ax, left + w / 2, top, label, size=SMALL)
        if values is not None:
            box(ax, left, y - h - 0.3, w, h, fill=color, edge=color, radius=0.6)
            text(ax, left + w / 2, y - h / 2 - 0.3, values[k], size=SMALL, color=on(color))


def s1d(ax):
    """scVI, Geneformer and scGPT: how a cell (or a bulk profile, treated as one cell) becomes an embedding; each row
    draws its model's own input encoding (170 x 53 mm)."""
    mm_axes(ax, 170, 53)
    rows = [  # model, color, how it was fit, input caption, bulk input, model box, per-cell output
        ("scVI", "scvi", "trained here,\nno labels", "counts per gene", "bulk: thinned to median cell depth",
         "variational\nencoder", "latent mean"),
        ("Geneformer", "geneformer", "pretrained,\nfrozen", "genes ranked by expression", "bulk: ranked as one cell",
         "Geneformer\ntransformer", "<cls>"),
        ("scGPT", "scgpt", "pretrained,\nfrozen", "gene tokens + binned values", "bulk: binned as one cell",
         "scGPT\ntransformer", "<cls>, normalized"),
    ]
    for r, (name, key, fit, caption, bulk_in, model, per_cell) in enumerate(rows):
        color = S.MODEL_COLORS[key]
        y = 43 - 17 * r                # row centre
        text(ax, 1, y + 2.2, name, size=BODY, color=color, ha="left", weight="bold")
        text(ax, 1, y - 3, fit, size=SMALL, ha="left")
        text(ax, 25, y + 5.8, caption, size=SMALL, ha="left")
        if key == "scvi":
            matrix(ax, 25, y - 1.75, 9, 1, 3.5, color, shades=np.random.default_rng(SEED).random((1, 9)))
        elif key == "geneformer":
            tokens(ax, 25, y, ["CLS", "G17", "G3", "G8", "…"], color)
        else:
            tokens(ax, 25, y, ["CLS", "G3", "G17", "G8", "…"], color, values=["", "5", "3", "2", ""])
        text(ax, 25, y - 6.3, bulk_in, size=SMALL, ha="left", color=GREY_DARK)
        arrow(ax, (59, y), (70, y))
        box(ax, 70, y - 5, 27, 10, fill=color)
        text(ax, 83.5, y, model, size=SMALL, color=on(color))
        arrow(ax, (97, y), (104, y))
        bar(ax, 104, y - 1.2, 14, 2.4, color)
        text(ax, 111, y - 4.6, per_cell, size=SMALL)
        arrow(ax, (118, y), (125, y))
        box(ax, 125, y - 5, 22, 10, fill="white")
        text(ax, 136, y, "mean over\ncells", size=SMALL)
        arrow(ax, (147, y), (153, y))
        bar(ax, 153, y - 1.2, 16, 2.4, color)
        text(ax, 161, y - 4.2, "embedding", size=SMALL)


def place(fig, rect, draw, item, panel):
    """Draw a schematic into a figure at rect (left, top, width, height in mm from the top-left corner) and write its CSV
    (the panel's text, one line per label: Additional file 3 has one file per panel); the caller places the letter."""
    width, height = fig.get_size_inches() * 25.4
    left, top, w, h = rect
    ax = fig.add_axes((left / width, 1 - (top + h) / height, w / width, h / height))
    draw(ax)
    common.panel_csv(pd.DataFrame({"text": [t.get_text().replace("\n", " ") for t in ax.texts]}), item, panel)


SIZES = {fig1a: (170, 45), fig1b: (88, 52), s1a: (170, 24), s1b: (85, 24), s1c: (85, 24), s1d: (170, 53)}   # mm


def main(out):
    """Previews: each panel alone at its print size."""
    style.apply()
    out.mkdir(parents=True, exist_ok=True)
    for draw, (width, height) in SIZES.items():
        fig = plt.figure(figsize=(width * MM, height * MM))
        draw(fig.add_axes((0, 0, 1, 1)))
        for suffix in (".pdf", ".png"):
            fig.savefig(out / f"{draw.__name__}{suffix}", dpi=600, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
