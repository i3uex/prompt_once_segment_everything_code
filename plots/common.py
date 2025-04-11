import logging

import matplotlib.pyplot as plt

from plots.constants import *

log = logging.getLogger(__name__)


def setup_pyplot():
    """
    Configure PyPlot not to use its default settings, so the output graph is
    more appealing.
    """

    log.info("Set PyPlot up")
    log.debug("setup_pyplot()")

    parameters = {"font.family": PYPLOT_FONT_FAMILY, "legend.fontsize": PYPLOT_LEGEND_FONT_SIZE,
                  "legend.facecolor": PYPLOT_LEGEND_FACE_COLOR, "figure.figsize": PYPLOT_FIGURE_SIZE,
                  "axes.labelsize": PYPLOT_AXES_LABEL_SIZE, "axes.titlesize": PYPLOT_AXES_TITLE_SIZE,
                  "grid.color": PYPLOT_GRID_COLOR, "xtick.labelsize": PYPLOT_XTICK_LABEL_SIZE,
                  "ytick.labelsize": PYPLOT_YTICK_LABEL_SIZE, "axes.titlepad": PYPLOT_AXES_TITLE_PAD,
                  "axes.edgecolor": PYPLOT_AXES_EDGE_COLOR, "axes.facecolor": PYPLOT_AXES_FACE_COLOR,
                  "xtick.color": PYPLOT_TICK_FONT_COLOR, "ytick.color": PYPLOT_TICK_FONT_COLOR,
                  "figure.facecolor": PYPLOT_FIGURE_FACE_COLOR, "axes.formatter.use_locale": True}

    log.debug(f"- parameters: {parameters}")

    plt.rcParams.update(parameters)
