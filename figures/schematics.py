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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

import common
import style
from sc2bulk.config import settings
from sc2bulk.config import style as S

MM = 1 / 25.4
INK = "#333333"                        # outlines, arrows and text
GREY = "#D9D9D9"                       # neutral fills (Fig 1A is greys only)
GREY_DARK = "#9E9E9E"
LIGHT = "#F2F2F2"                      # discarded heads
SECONDARY, SECONDARY_FILL = "#707070", "#B5B5B5"   # Fig 1A's pseudobulk row: legible, lighter than the samples
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


def matrix(ax, x, y, ncol, nrow, cell, fill, edge="white", shades=None, outline=INK):
    """A cell x gene grid (cells are rows, genes columns); shades (nrow x ncol, 0-1) vary the fill's lightness, as
    BioRender's heat matrices do."""
    for i in range(nrow):
        for j in range(ncol):
            alpha = 1.0 if shades is None else 0.35 + 0.65 * shades[i, j]
            ax.add_patch(Rectangle((x + j * cell, y + i * cell), cell, cell, facecolor=fill, alpha=alpha,
                                   edgecolor="none"))
    for k in range(ncol + 1):
        ax.plot([x + k * cell] * 2, [y, y + nrow * cell], color=edge, linewidth=0.5)
    for k in range(nrow + 1):
        ax.plot([x, x + ncol * cell], [y + k * cell] * 2, color=edge, linewidth=0.5)
    ax.add_patch(Rectangle((x, y), ncol * cell, nrow * cell, facecolor="none", edgecolor=outline, linewidth=0.4))


def bar(ax, x, y, w, h, fill, edge=INK):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=0.4))


def dot(ax, x, y, filled, r=0.9):
    ax.add_patch(Circle((x, y), r, facecolor=INK if filled else "white", edgecolor=INK, linewidth=0.5))


B_ROW_TOP, B_ROW_PITCH = 39.3, 10.6   # Fig 1B's rows: the first row's centre (mm from the bottom) and the pitch
GENES, CELLS = 9, 6                    # the drawn sample: a cells x genes matrix (Fig 1A and each row of Fig 1B)
REPRESENTATION = 5                     # the drawn sample representation's length


def sample_shades():
    return np.random.default_rng(SEED).random((CELLS, GENES))


def fig1a(ax):
    """The task: a single-cell sample (cells x genes), its pseudobulk (the cells summed after sequencing) and a bulk
    sample (summed before sequencing) on one gene axis; one model turns each sample into a sample representation; a
    ridge readout trained on single-cell samples is applied unchanged to bulk samples and scored by bulk AUROC
    (170 x 45 mm, greys only)."""
    mm_axes(ax, 170, 45)
    shades = sample_shades()
    profile = shades.mean(axis=0, keepdims=True)
    profile = (profile - profile.min()) / (profile.max() - profile.min())
    x0, cell = 33, 2.0                 # the gene columns, shared by the three rows
    mid = x0 + GENES * cell / 2
    right = x0 + GENES * cell
    text(ax, mid, 43.6, f"{settings.PANEL_SIZE:,} genes →", size=SMALL)
    # the single-cell sample: cells x genes
    matrix(ax, x0, 29.5, GENES, CELLS, cell, GREY_DARK, shades=shades)
    text(ax, x0 - 1.6, 35.5, "cells", size=SMALL, rotation=90)
    text(ax, 1, 35.5, "Single-cell sample", ha="left", weight="bold")
    # its pseudobulk, faded: it explains bulk and is not a step of the task
    arrow(ax, (mid, 29), (mid, 23.4), color=SECONDARY)
    text(ax, mid + 1.2, 26.2, "Σ", size=SMALL, color=SECONDARY, ha="left")
    matrix(ax, x0, 21, GENES, 1, cell, SECONDARY_FILL, shades=profile, outline=SECONDARY)
    text(ax, 1, 22, "Pseudobulk", ha="left", weight="bold", color=SECONDARY)
    text(ax, 1, 17.9, "sum over cells,\nafter sequencing", size=SMALL, ha="left", color=SECONDARY)
    ax.plot([1, right], [14.2, 14.2], color=INK, linewidth=LINE)
    # the bulk sample
    matrix(ax, x0, 9, GENES, 1, cell, GREY_DARK, shades=np.clip(profile + 0.1 * np.random.default_rng(SEED + 1)
                                                                .standard_normal(profile.shape), 0, 1))
    text(ax, 1, 10, "Bulk sample", ha="left", weight="bold")
    text(ax, 1, 5.9, "sum over cells,\nbefore sequencing", size=SMALL, ha="left")
    # one model, shared by both samples: wide in, narrow out
    arrow(ax, (right + 0.6, 35.5), (55, 35.5))
    arrow(ax, (right + 0.6, 10), (55, 10))
    ax.add_patch(Polygon([(55, 5), (55, 41), (66, 29), (66, 15)], closed=True, facecolor=GREY, edgecolor=INK,
                         linewidth=LINE, joinstyle="round"))
    text(ax, 60.5, 23.3, "Model", weight="bold")
    text(ax, 60.5, 20.3, "(b)", size=SMALL)
    # the two sample representations
    rep = np.linspace(1, 0, REPRESENTATION)
    for y, order, label, va, dy in ((27, rep, "single-cell sample\nrepresentation", "bottom", 2.0),
                                    (17, rep[::-1], "bulk sample\nrepresentation", "top", -2.0)):
        arrow(ax, (66, y), (69.5, y))
        matrix(ax, 70, y - 1, REPRESENTATION, 1, 2.0, GREY_DARK, shades=order[None, :])
        text(ax, 75, y + dy, label, size=SMALL, va=va)
    # the ridge readout: trained on the single-cell representations, applied unchanged to the bulk ones
    arrow(ax, (80.5, 27), (100, 27))
    text(ax, 90, 27.8, "train", size=SMALL, va="bottom")
    arrow(ax, (80.5, 17), (100, 17))
    text(ax, 89.9, 17.8, "apply unchanged", size=SMALL, va="bottom")
    box(ax, 100, 14, 18, 16)
    text(ax, 109, 22, "Ridge\nclassifier")
    text(ax, 109, 43.3, "single-cell labels", size=SMALL)
    dot(ax, 106.8, 39.7, False)
    dot(ax, 111.2, 39.7, True)
    arrow(ax, (109, 38.3), (109, 30))
    # the predicted bulk label, scored by bulk AUROC
    arrow(ax, (118, 22), (124, 22))
    box(ax, 124, 13, 26, 18)
    text(ax, 137, 34, "Predicted bulk label")
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
    """How each family turns a single-cell sample into a sample representation; every row draws the same matrix, model
    box and representation (88 x 52 mm, family colors)."""
    mm_axes(ax, 88, 52)
    shades = sample_shades()
    rep = np.linspace(1, 0, REPRESENTATION)[None, :]
    cell = 1.4                         # Fig 1A's matrix, scaled
    x0 = 27.5
    mid = x0 + GENES * cell / 2
    box_x, box_w = x0 + GENES * cell + 3.5, 25
    out_x = box_x + box_w + 3.5
    out_cell = 2.6
    out_mid = out_x + REPRESENTATION * out_cell / 2
    text(ax, mid, 50.8, "Single-cell sample", weight="bold")
    text(ax, mid, 45.3, f"{settings.PANEL_SIZE:,} genes →", size=SMALL)
    text(ax, out_mid, 49.3, "Sample\nrepresentation", weight="bold")
    rows = [  # family, family label, model box
        ("pseudobulk", "Pseudobulk", "Σ sum"),
        ("set_encoder", "Supervised set\nencoders", "CytoSet, Set\nTransformer, ScRAT"),
        ("generative", "Unsupervised\ngenerative model", "scVI"),
        ("foundation_model", "Frozen foundation\nmodels", "Geneformer, scGPT"),
    ]
    for r, (key, family, model) in enumerate(rows):
        color = S.FAMILY_COLORS[key]
        y = B_ROW_TOP - B_ROW_PITCH * r   # row centre
        text(ax, 0.5, y, family, size=BODY, color=color, ha="left", weight="bold")
        matrix(ax, x0, y - CELLS * cell / 2, GENES, CELLS, cell, color, shades=shades)
        if r == 0:
            text(ax, x0 - 1.4, y, "cells", size=SMALL, rotation=90)
        arrow(ax, (x0 + GENES * cell + 0.5, y), (box_x, y))
        box(ax, box_x, y - 4, box_w, 8, fill=color)
        text(ax, box_x + box_w / 2, y, model, size=SMALL, color="white")   # white: the best worst-case contrast on the four fills
        arrow(ax, (box_x + box_w, y), (out_x - 0.5, y))
        matrix(ax, out_x, y - out_cell / 2, REPRESENTATION, 1, out_cell, color, shades=rep)


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
