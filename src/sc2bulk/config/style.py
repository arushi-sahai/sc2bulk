"""The figure style every figure script imports. One
encoding per meaning, never reused. Configuration colors are MODEL_COLORS below (by model); pairs are shown by panel
position and are never color-coded. Imported by no pipeline stage (tests/test_codehash.py)."""

# Family colors (keyed by configurations.MODELS "family"): the paper-wide Okabe-Ito family
# colors. Used only for families (Fig 1B's rows, Fig 1C's row tints and family labels); each
# model's color (MODEL_COLORS, also Fig. S1's encoders) is a shade of its family's
# (tests/test_palette.py). Phenotype groups stay neutral.
FAMILY_COLORS = {"pseudobulk": "#0072B2", "set_encoder": "#D55E00", "generative": "#009E73", "foundation_model": "#CC79A7"}
FAMILY_TINT = 0.15                     # alpha of a family's row tint
# Model colors, keyed by configurations.MODELS (one hue per family, the models differing in lightness): the figures' single source. configurations.MODELS' own "color" field is stale and never read by a figure
# (tests/test_figures_config.py): configurations.py is in stage 05's code hash and every stage's from 08 on, so it stays
# unchanged.
MODEL_COLORS = {"pseudobulk": "#0072B2", "cytoset": "#D55E00", "set_transformer": "#8C3A00", "scrat": "#F0A35E",
                "scvi": "#009E73", "geneformer": "#A3487F", "scgpt": "#CC79A7"}

# The color rule: configuration figures (Fig 1C, Figs 2-6, Figs. S9–S16) use only the
# family and configuration colors; descriptive figures (Fig 1D/E, Figs. S2–S8) never show configurations and use only
# the data colors below (Paul Tol's muted scheme): the side (single-cell vs. bulk) and the cell type. Phenotype groups
# are open (0) vs. filled (1) everywhere, never a color.
SIDE_COLORS = {"single_cell": "#DDCC77", "bulk": "#332288"}   # Tol muted sand and indigo
# Cell types: one color per type name, the same in every pair and figure (figures/style.cell_type_colors); a pair's
# types beyond the palette's nine are grouped as "other". "other" is lightened from Tol's #BBBBBB to #C9C9C9 so it
# stays >= 15 CIELAB from teal under deutan/protan simulation (tests/test_palette.py).
CELL_TYPE_PALETTE = ("#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499")
OTHER_TYPE = {"name": "other", "color": "#C9C9C9"}
GROUP_FILL = {0: False, 1: True}       # open = 0 (younger / control), filled = 1 (older / disease)
# The rules that keep each encoding to one meaning (asserted where cheap: tests/test_palette.py, tests/test_figures_config.py):
#   - open vs. filled means the phenotype group only in figures without configuration markers (Fig 1d, 3a, Figs. S2–S8, S13a–e);
#     where configuration markers and groups share a panel (Fig. S11a's per-sample strips) the group is shown by x-position or
#     facet, never by fill (hollow there means pseudobulk input);
#   - data colors (SIDE_COLORS, CELL_TYPE_PALETTE) and family hues (FAMILY_COLORS, MODEL_COLORS) never share a panel;
#   - the pseudobulk reference line is dashed, in pseudobulk's color #0072B2 (PSEUDOBULK_LINE).
assert not set(SIDE_COLORS.values()) & {*FAMILY_COLORS.values(), *MODEL_COLORS.values()}

# Pattern = input: pseudobulk-input configurations are hatched (bars) or hollow (markers)
PSEUDOBULK_HATCH = "////"
MARKER_EDGE = 0.6                      # a marker's outline, pt (>= 0.5 pt, checklist)
# A hollow mean marker: a white face over the lines and dots behind it, a heavier outline and a little larger, so the
# hollow reads at print size; hollow seed dots stay open with the thin outline
HOLLOW_MARKER = {"facecolor": "white", "linewidth": 1.0, "size_factor": 1.12}

# Alignment condition = a labeled x-offset within each configuration's group (Fig 3b/c and its supplement), the
# conditions spread evenly over this width in their figure's order and joined by a thin line in the configuration's
# color; no shape glyphs, so model markers keep one meaning
ALIGNMENT_SPREAD = 0.6
TRACE_WIDTH = 0.5                      # the line joining a configuration's conditions

# Samples (Fig 3a, Fig. S13a–e; reviewer fixes adopted 2026-10-08): one circle shape, side = SIDE_COLORS, group = filled (1)
# vs. open (0) (figures/style.modality); the rows are labelled in text, no family hues in those panels
SAMPLE_MARKER = "o"
MODALITY_COLOR = "#33323a"             # neutral black for non-data marks (dots of a characteristic grid, connectors)
GROUP_X_OFFSET = 0.35                  # where configurations and groups share a panel (Fig. S11a): group by x-position

# The selected configuration = a ring drawn around its point, visibly larger than a hollow
# marker so it never reads as "pseudobulk input"; legend entry "selected"
SELECTED_RING = {"marker": "o", "facecolor": "none", "linewidth": 1.0, "size_factor": 2.0, "label": "selected"}

# Reference rows (not configurations): black, their own glyph
REFERENCE_STYLE = {"color": "#000000", "marker": "X"}
REFERENCE_GRAY = "#808080"             # Fig 2a's reference and control rows: mid gray, set apart from the nine configurations
REFERENCE_RULE = {"color": "#9e9e9e", "linewidth": 0.5}   # Fig 2a: the thin rule above the reference rows

# Reference lines; where the within-bulk reference and chance appear together, both are
# labeled directly
# The dashed pseudobulk-blue line = 0 on every difference-from-pseudobulk axis (incl. Fig 4A, where the
# n-cell pseudobulk is a data series: solid, in pseudobulk's color and marker)
PSEUDOBULK_LINE = {"linestyle": "--", "linewidth": 1.0, "color": MODEL_COLORS["pseudobulk"]}
WITHIN_BULK_LINE = {"linestyle": "-.", "linewidth": 1.0, "color": "#4d4d4d"}
CHANCE_LINE = {"linestyle": ":", "linewidth": 1.0, "color": "#4d4d4d"}
MARGIN_BAND = {"color": "#e6e6e6", "linewidth": 0}       # Fig 2A and Fig. S9: from +0.05 to the axis edge
DIVIDER = {"linestyle": ":", "linewidth": 0.5, "color": "#4d4d4d"}   # Fig 4: candidates | the ring point

# Values that hold by construction (Fig 5D's simulation): open gray marks, references, never results
BY_CONSTRUCTION_COLOR = "#9e9e9e"

# Seeds as faint dots with a mean marker
SEED_DOT_ALPHA = 0.4
MARKER_SIZE = 3.0                      # points (the mean marker); seed dots are SEED_DOT_FACTOR of it (layout pass,
SEED_DOT_FACTOR = 0.45                 # 2026-10-09: 4.0 / 0.6 / alpha 0.3 piled up; checklist: means ~3 pt, seeds ~1.3 pt)
LINE_WIDTH = 0.75                      # data lines and CI bars
MIN_LINE_WIDTH_PT = 0.25               # Genome Biology: every line wider than 0.25 pt at final size (asserted)
LEGEND_CHARS_PER_LINE = 120            # a typeset legend at the journal's ~8 pt over 170 mm (an estimate for the height rule)
LEGEND_LINE_MM = 3.4                   # its line height
# Layout checks (figures/common.check_layout), measured in mm on the rendered figure
LAYOUT_LETTER_ALIGN_MM = 1.0           # L1: the left column's panel letters share a left edge within this
LAYOUT_LETTER_TOP_MM = (5.0, 3.0)      # L2: a letter's top at most this far above (in the row gap) / below (beside a multi-line header) its panel's highest text or axes
LAYOUT_ROW_GAP_MM = (4.0, 7.0)         # W1: the gap between consecutive rows' artists
LAYOUT_EMPTY_MM = 8.0                  # W2: no empty band or side margin wider; W3: no gap this wide inside a panel
LAYOUT_OVERLAP_MM = 0.2                # X2: boxes overlapping by more than this, both ways, overlap
LAYOUT_COLUMN_ALIGN_MM = 1.0           # X3: matching pair columns' left edges across rows
LAYOUT_SIBLING_MM = 0.5                # X4: sibling axes' tops and heights
LAYOUT_TEXT_PT = (6, 8)                # X5: text other than tick labels and panel letters
LAYOUT_YLABEL_GAP_MM = 3.0             # X6: a y label's distance from its axes' tick labels
LAYOUT_XLABEL_CENTER_MM = 2.0          # X6: a shared x label's offset from the center of its columns

# Grayscale for Fig 6A's heatmap (0-1); the one diverging map is Fig. S16's, centered at 0
HEATMAP_CMAP = "Greys"
DIVERGING_CMAP = "RdBu_r"
NA_HATCH = {"hatch": "xxxx", "facecolor": "none", "edgecolor": "#9e9e9e", "linewidth": 0.3}   # not-applicable cells

# Panels where all seven configurations appear as same-shape dots label them directly
DIRECT_LABELS = True

# Genome Biology (fetched 2026-10-07): 85 or 170 mm wide, <= 225 mm tall including the legend, ~300 dpi,
# fonts embedded (TrueType: fonttype 42), <= 10 MB per figure file, <= 20 MB per additional file
FONT = "Arial"
FONT_FALLBACKS = ("Helvetica", "Liberation Sans", "Nimbus Sans")
FONT_SIZE = 8
MIN_FONT_SIZE = 6                      # tick labels and annotations (house rule; the journal asks only "legible")
PANEL_LABEL = {"size": 10, "weight": "bold"}
WIDTH_MM = {"single": 85, "double": 170}
MAX_HEIGHT_MM = 225
DPI = 300
RASTER_DPI = 600                       # rasterized artists inside the PDF and SVG (dense scatters), sharp when enlarged
FILE_FORMAT = "pdf"
PDF_FONTTYPE = 42
MAX_FIGURE_BYTES = 10 * 2**20
MAX_ADDITIONAL_FILE_BYTES = 20 * 2**20
