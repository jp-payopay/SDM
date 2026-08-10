import os

# Output resolution for every saved figure (raster maps, response curves,
# variable-importance bars). Defaults to 1200 dpi, a common publication
# requirement for raster figures and line art. Figures are laid out in inches
# (figsize) with point-based fonts, so raising only the save dpi scales
# everything up proportionally and stays crisp. At these figsizes 1200 dpi
# yields large PNGs (a 5x5 in map is 6000x6000 px); the HTML report displays
# them scaled down, but the on-disk files are full publication resolution.
#
# Override with the SDM_FIGURE_DPI environment variable to render smaller,
# faster figures during development or testing (the test suite sets it low).
PUBLICATION_DPI = int(os.environ.get("SDM_FIGURE_DPI", "1200"))
