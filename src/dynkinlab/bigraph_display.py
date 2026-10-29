# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Deterministic inline SVG rendering for mixed bigraphs."""

import io

import networkx as nx
from matplotlib import rc_context
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

from dynkinlab.bigraph import Bigraph, EdgeStyle
from dynkinlab.display import DisplayOptions


class BigraphRenderer:
    """Render a bigraph as a self-contained SVG fragment."""

    @classmethod
    def render(cls, bigraph: Bigraph, options: DisplayOptions) -> str:
        """Render the bigraph as an inline SVG string.

        Parameters
        ----------
        bigraph : Bigraph
            The bigraph to render.
        options : DisplayOptions
            Resolved display options.

        Returns
        -------
        str
            An SVG fragment beginning at the ``<svg>`` element.
        """
        graph = nx.Graph()
        graph.add_nodes_from(bigraph.labels)
        for edge in bigraph.edges:
            graph.add_edge(edge.source, edge.target)

        if bigraph.dimension == 1:
            positions = {bigraph.labels[0]: (0.0, 0.0)}
        else:
            positions = nx.spring_layout(
                graph,
                seed=options.bigraph_layout_seed,
            )

        figure = Figure(figsize=options.bigraph_figure_size, dpi=100)
        FigureCanvasSVG(figure)
        axes = figure.add_subplot(111)
        axes.set_aspect("equal")
        axes.axis("off")

        for label, (x_coordinate, y_coordinate) in positions.items():
            axes.scatter(
                [x_coordinate],
                [y_coordinate],
                s=240,
                c="white",
                edgecolors="black",
                zorder=3,
            )
            if options.bigraph_show_labels:
                axes.text(
                    x_coordinate,
                    y_coordinate,
                    str(label),
                    ha="center",
                    va="center",
                    fontsize=10,
                    zorder=4,
                )

        for edge in bigraph.edges:
            start = positions[edge.source]
            end = positions[edge.target]
            linestyle = ":" if edge.style is EdgeStyle.DOTTED else "-"
            if edge.directed:
                arrowstyle = "-|>" if options.bigraph_arrow_style == "arrow" else "-"
                patch = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle=arrowstyle,
                    mutation_scale=12,
                    linewidth=1.2,
                    linestyle=linestyle,
                    color="black",
                    shrinkA=10,
                    shrinkB=10,
                    zorder=2,
                )
                axes.add_patch(patch)
            else:
                axes.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    linestyle=linestyle,
                    linewidth=1.2,
                    color="black",
                    zorder=1,
                )
            if options.bigraph_show_multiplicity and edge.multiplicity > 1:
                axes.text(
                    (start[0] + end[0]) / 2,
                    (start[1] + end[1]) / 2,
                    str(edge.multiplicity),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    zorder=5,
                )

        buffer = io.StringIO()
        with rc_context({"svg.hashsalt": "dynkinlab"}):
            figure.savefig(
                buffer,
                format="svg",
                bbox_inches="tight",
                transparent=True,
                metadata={"Date": None},
            )
        svg = buffer.getvalue()
        start = svg.find("<svg")
        return svg[start:] if start >= 0 else svg
