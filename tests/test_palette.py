"""The configuration palette (config/style.MODEL_COLORS) stays distinguishable under the three common color-vision
deficiencies (Machado et al. 2009 simulation, severity 1) and in normal vision: every pair of
model colors differs by at least 15 in CIELAB. Grayscale is covered by the marker shapes,
which must all differ. The descriptive figures' data colors (the two sides; the cell-type palette with "other")
pass the same check. One hue per family: every model's color is a shade of its family's
color (config/style.FAMILY_COLORS)."""
import colorsys
import itertools

import numpy as np

from sc2bulk.config.configurations import MODELS
from sc2bulk.config.style import CELL_TYPE_PALETTE, FAMILY_COLORS, MODEL_COLORS, OTHER_TYPE, REFERENCE_STYLE, SIDE_COLORS

MACHADO = {
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}
MIN_DELTA_E = 15
MAX_HUE_DEGREES = 25            # a model's hue vs. its family's


def linear_rgb(color):
    c = np.array([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lab(rgb):
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ np.clip(rgb, 0, 1)
    t = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def distinguishable(named):
    """Every pair of named colors differs by at least MIN_DELTA_E in normal vision and each simulated deficiency."""
    for condition in ("normal", *MACHADO):
        colors = {}
        for name, color in named.items():
            rgb = linear_rgb(color)
            colors[name] = lab(rgb if condition == "normal" else np.array(MACHADO[condition]) @ rgb)
        for a, b in itertools.combinations(colors, 2):
            assert np.linalg.norm(colors[a] - colors[b]) >= MIN_DELTA_E, f"{condition}: {a} vs {b}"


def test_palette_survives_color_vision_deficiencies():
    assert set(MODEL_COLORS) == set(MODELS)
    distinguishable(MODEL_COLORS)


def test_data_colors_survive_color_vision_deficiencies():
    distinguishable(SIDE_COLORS)
    distinguishable({**{c: c for c in CELL_TYPE_PALETTE}, OTHER_TYPE["name"]: OTHER_TYPE["color"]})


def test_markers_are_unique():
    markers = [spec["marker"] for spec in MODELS.values()] + [REFERENCE_STYLE["marker"]]
    assert len(markers) == len(set(markers))


def hue(color):
    return 360 * colorsys.rgb_to_hsv(*(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)))[0]


def test_models_are_shades_of_their_family():
    for model, spec in MODELS.items():
        gap = abs(hue(MODEL_COLORS[model]) - hue(FAMILY_COLORS[spec["family"]]))
        assert min(gap, 360 - gap) <= MAX_HUE_DEGREES, model
