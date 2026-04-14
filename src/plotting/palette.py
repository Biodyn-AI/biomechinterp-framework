"""Colorblind-safe color palette helper.

Reviewer R3.3 flagged the red/green covariate-balance scheme in Figure 6 as
inaccessible to readers with Protanopia/Deuteranopia. We standardise on the
Okabe-Ito 8-color qualitative palette, which is one of the most widely
recommended colorblind-safe palettes (originally proposed by Okabe and Ito,
2008; popularised in scientific visualisation by colorbrewer and many
plotting libraries).

Use these named constants instead of literal hex codes when adding or
modifying figures so that the palette stays consistent across all panels.

Reference Okabe-Ito values:
  black           '#000000'
  orange          '#E69F00'
  sky_blue        '#56B4E9'
  bluish_green    '#009E73'
  yellow          '#F0E442'
  blue            '#0072B2'
  vermillion      '#D55E00'
  reddish_purple  '#CC79A7'

Recommended pairings to replace the legacy red/green pair:
  ``BEFORE`` (was red C44E52)   -> ``vermillion`` (#D55E00)
  ``AFTER``  (was green 55A868) -> ``blue``       (#0072B2)
These two are well separated for both Protanopia and Deuteranopia.
"""

OKABE_ITO = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Common semantic aliases used throughout figures
PALETTE = {
    "before":      OKABE_ITO["vermillion"],   # was '#C44E52'
    "after":       OKABE_ITO["blue"],         # was '#55A868'
    "attn":        OKABE_ITO["blue"],         # was '#4C72B0'
    "corr":        OKABE_ITO["orange"],       # was '#E8A838'
    "gene_only":   OKABE_ITO["bluish_green"], # was '#55A868'
    "gene_attn":   OKABE_ITO["reddish_purple"],
    "gene_corr":   OKABE_ITO["yellow"],
    "trrust":      OKABE_ITO["vermillion"],
    "composite":   OKABE_ITO["bluish_green"],
    "random":      OKABE_ITO["sky_blue"],
    "baseline":    OKABE_ITO["black"],
}


def get(name: str, default: str = OKABE_ITO["black"]) -> str:
    """Look up a semantic color, falling back to the requested default."""
    return PALETTE.get(name, default)
