# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 09:16:56 2015

@author: adelpret
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

DEFAULT_FONT_SIZE = 20
DEFAULT_AXIS_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_LINE_WIDTH = 4  # 13
DEFAULT_MARKER_SIZE = 6
DEFAULT_FONT_FAMILY = "sans-serif"
DEFAULT_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_FONT_SERIF = [
    "Times New Roman",
    "Times",
    "Bitstream Vera Serif",
    "DejaVu Serif",
    "New Century Schoolbook",
    "Century Schoolbook L",
    "Utopia",
    "ITC Bookman",
    "Bookman",
    "Nimbus Roman No9 L",
    "Palatino",
    "Charter",
    "serif",
]
DEFAULT_FIGURE_FACE_COLOR = "white"  # figure facecolor 0.75 is scalar gray
DEFAULT_LEGEND_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_AXES_LABEL_SIZE = DEFAULT_FONT_SIZE  # fontsize of the x any y labels
DEFAULT_TEXT_USE_TEX = False
LINE_ALPHA = 0.9
SAVE_FIGURES = True
FILE_EXTENSIONS = ["png"]  # 'pdf'] #['png'] #,'eps']
FIGURES_DPI = 150
SHOW_LEGENDS = False
LEGEND_ALPHA = 0.5
SHOW_FIGURES = False
FIGURE_PATH = "./"
LINE_WIDTH_RED = 0  # reduction of line width when plotting multiple lines on same plot
LINE_WIDTH_MIN = 1
BOUNDS_COLOR = "silver"

# legend.framealpha    : 1.0    # opacity of of legend frame
# axes.hold           : True    # whether to clear the axes by default on
# axes.linewidth      : 1.0     # edge linewidth
# axes.titlesize      : large   # fontsize of the axes title
# axes.color_cycle    : b, g, r, c, m, y, k  # color cycle for plot lines
# xtick.labelsize      : medium # fontsize of the tick labels
# figure.dpi       : 80      # figure dots per inch
# image.cmap   : jet               # gray | jet etc...
# savefig.dpi         : 100      # figure dots per inch
# savefig.facecolor   : white    # figure facecolor when saving
# savefig.edgecolor   : white    # figure edgecolor when saving
# savefig.format      : png      # png, ps, pdf, svg
# savefig.jpeg_quality: 95       # when a jpeg is saved, the default quality parameter.
# savefig.directory   : ~        # default directory in savefig dialog box,
#                                # leave empty to always use current working directory


def create_empty_figure(nRows=1, nCols=1, figsize=(7, 7), spinesPos=None, sharex=True):
    f, ax = plt.subplots(nRows, nCols, figsize=figsize, sharex=sharex)
    mngr = plt.get_current_fig_manager()
    # mngr.window.setGeometry(50,50,1080,720)

    if spinesPos is not None:
        if nRows * nCols > 1:
            for axis in ax.reshape(nRows * nCols):
                movePlotSpines(axis, spinesPos)
        else:
            movePlotSpines(ax, spinesPos)
    return f, ax


def movePlotSpines(ax, spinesPos):
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["bottom"].set_position(("data", spinesPos[0]))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("data", spinesPos[1]))


def setAxisFontSize(ax, size):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(size)
        label.set_bbox(dict(facecolor="white", edgecolor="None", alpha=0.65))


mpl.rcdefaults()
mpl.rcParams["lines.linewidth"] = DEFAULT_LINE_WIDTH
mpl.rcParams["lines.markersize"] = DEFAULT_MARKER_SIZE
mpl.rcParams["patch.linewidth"] = 1
mpl.rcParams["axes.grid"] = True
mpl.rcParams["font.family"] = DEFAULT_FONT_FAMILY
mpl.rcParams["font.size"] = DEFAULT_FONT_SIZE
mpl.rcParams["font.serif"] = DEFAULT_FONT_SERIF
mpl.rcParams["text.usetex"] = DEFAULT_TEXT_USE_TEX
mpl.rcParams["axes.labelsize"] = DEFAULT_AXES_LABEL_SIZE
mpl.rcParams["legend.fontsize"] = DEFAULT_LEGEND_FONT_SIZE
mpl.rcParams["figure.facecolor"] = DEFAULT_FIGURE_FACE_COLOR
mpl.rcParams["figure.figsize"] = 14 , 7 #23, 12  # 12, 9 #


def saveFigure(title):
    if SAVE_FIGURES:
        for ext in FILE_EXTENSIONS:
            plt.gcf().savefig(
                FIGURE_PATH + title.replace(" ", "_") + "." + ext,
                format=ext,
                dpi=FIGURES_DPI,
                bbox_inches="tight",
            )
