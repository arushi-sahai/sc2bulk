"""Fig. S1: the encoders' architectures (figures/schematics.py; label-free). A: CytoSet across the top; B and C: Set
Transformer and ScRAT side by side; D: scVI, Geneformer and scGPT, how a cell or a bulk profile becomes an embedding."""
import common
import schematics
import style

FIGURE = "Fig. S1"
READS_BULK_LABELS = False
READS_SCORES = False
# Each panel at print size: (left, top, width, height) in mm from the composite's top-left corner, and its drawing
PANELS = {"A": ((0, 4, *schematics.SIZES[schematics.s1a]), schematics.s1a),
          "B": ((0, 32, *schematics.SIZES[schematics.s1b]), schematics.s1b),
          "C": ((85, 32, *schematics.SIZES[schematics.s1c]), schematics.s1c),
          "D": ((0, 58.5, *schematics.SIZES[schematics.s1d]), schematics.s1d)}
# Each letter's top in mm from the composite's top: 2-4 mm above its panel's highest drawing (the drawings start below
# their rects' tops by different amounts)
LETTER_TOP = {"A": 0, "B": 30, "C": 30, "D": 58}


def panel(letter):
    rect, draw = PANELS[letter]

    def drawn(fig, cell, pairs):
        schematics.place(fig, rect, draw, FIGURE, letter)
        style.panel_label_mm(fig, letter, rect[0], LETTER_TOP[letter])   # at the panel's left edge
    return drawn


def main(pairs, panels):
    common.draw(FIGURE, {letter: panel(letter) for letter in PANELS}, pairs, panels, __file__, rows=(("A",), ("B", "C"), ("D",)))
