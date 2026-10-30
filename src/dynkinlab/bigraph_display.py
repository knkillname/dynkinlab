# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""NetworkX layouts and matplotlib rendering for mixed bigraphs and block-cut trees."""

import io
from collections.abc import Hashable, Iterable, Mapping

import networkx as nx
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

from dynkinlab import display
from dynkinlab.bigraph import Bigraph, EdgeStyle
from dynkinlab.display import DisplayOptions
from dynkinlab.structure import BlockCutTree

_LAYOUTS = frozenset({"auto", "planar", "kamada_kawai", "spring", "circular"})
_HIGHLIGHT_EDGE = "#b03a2e"
_EDGE_COLOR = "#34495e"
_HIGHLIGHT_NODE = "#f7dc6f"
_NODE_COLOR = "#ecf0f1"
_BLOCK_COLOR = "#aed6f1"
_TREE_EDGE = "#7f8c8d"
_NODE_SIZE = 520
_BLOCK_SIZE = 480
_MULTIPLICITY_OFFSET = 0.11
_BLOCK_LABEL_OFFSET = 0.22
_SINGLE_VERTEX_POSITION = (0.0, 0.0)
_K4_CLIQUE_SIZE = 4
_FALLBACK_SPRING_ITERATIONS = 200


def _validate_seed(seed: int) -> None:
    """Reject invalid layout seeds."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _validate_iterations(iterations: int) -> None:
    """Reject invalid spring iteration counts."""
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise ValueError("iterations must be a positive integer")
    if iterations <= 0:
        raise ValueError("iterations must be a positive integer")


def _contains_k4(graph: "nx.Graph[Hashable]") -> bool:
    """Return whether the graph contains a four-vertex clique."""
    return any(len(clique) >= _K4_CLIQUE_SIZE for clique in nx.find_cliques(graph))


def _spring_positions(
    graph: "nx.Graph[Hashable]",
    *,
    seed: int,
    iterations: int,
    pos: Mapping[Hashable, tuple[float, float]] | None = None,
) -> dict[Hashable, tuple[float, float]]:
    """Run the energy-based spring layout with unweighted edges."""
    try:
        raw = nx.spring_layout(
            graph,
            pos=None if pos is None else dict(pos),
            seed=seed,
            iterations=iterations,
            weight=None,
            method="energy",
        )
    except ImportError:
        raw = nx.spring_layout(
            graph,
            pos=None if pos is None else dict(pos),
            seed=seed,
            iterations=min(iterations, _FALLBACK_SPRING_ITERATIONS),
            weight=None,
            method="force",
        )
    return {label: (float(position[0]), float(position[1])) for label, position in raw.items()}


def compute_layout(
    bigraph: Bigraph,
    layout: str = "auto",
    seed: int = 0,
    iterations: int = 10000,
) -> dict[Hashable, tuple[float, float]]:
    """Compute vertex coordinates with NetworkX.

    The automatic policy prefers Kamada-Kawai for connected graphs when SciPy
    is available. A planar graph containing a four-vertex clique starts from a
    planar embedding and is refined by the energy-based spring layout. Other
    graphs fall back to the same energy-based spring layout. Layout distances
    ignore the bigraph's signed edge weights and use unweighted adjacency.

    Parameters
    ----------
    bigraph : Bigraph
        Bigraph whose labels become graph nodes.
    layout : str
        Layout policy: ``"auto"``, ``"planar"``, ``"kamada_kawai"``,
        ``"spring"``, or ``"circular"``.
    seed : int
        Seed used by the spring layout.
    iterations : int
        Iteration count used by the spring layout.

    Returns
    -------
    dict[Hashable, tuple[float, float]]
        One coordinate pair for every label.

    Raises
    ------
    ValueError
        If the layout policy, seed, or iteration count is invalid.
    """
    if layout not in _LAYOUTS:
        raise ValueError(f"unknown bigraph layout: {layout!r}")
    _validate_seed(seed)
    _validate_iterations(iterations)
    if bigraph.dimension == 1:
        return {bigraph.labels[0]: _SINGLE_VERTEX_POSITION}

    graph = bigraph.to_networkx()
    if layout == "auto":
        if nx.is_connected(graph):
            if _contains_k4(graph) and nx.check_planarity(graph)[0]:
                planar = {
                    label: (float(position[0]), float(position[1]))
                    for label, position in nx.planar_layout(graph).items()
                }
                positions = _spring_positions(
                    graph,
                    seed=seed,
                    iterations=iterations,
                    pos=planar,
                )
            else:
                try:
                    positions = nx.kamada_kawai_layout(graph, weight=None)
                except ImportError:
                    positions = _spring_positions(
                        graph,
                        seed=seed,
                        iterations=iterations,
                    )
        else:
            positions = _spring_positions(
                graph,
                seed=seed,
                iterations=iterations,
            )
    elif layout == "planar":
        positions = nx.planar_layout(graph)
    elif layout == "kamada_kawai":
        positions = nx.kamada_kawai_layout(graph, weight=None)
    elif layout == "spring":
        positions = _spring_positions(
            graph,
            seed=seed,
            iterations=iterations,
        )
    else:
        positions = nx.circular_layout(graph)
    return {
        label: (float(position[0]), float(position[1])) for label, position in positions.items()
    }


def _draw_bigraph_edges(
    ax: Axes,
    bigraph: Bigraph,
    points: Mapping[Hashable, tuple[float, float]],
    highlighted_edges: set[frozenset[Hashable]],
    *,
    show_multiplicity: bool,
    arrow_style: str,
) -> None:
    """Draw the edges and arrows of a bigraph on an axes."""
    for edge in bigraph.edges:
        first = points[edge.source]
        second = points[edge.target]
        pair = frozenset((edge.source, edge.target))
        highlighted = pair in highlighted_edges
        color = _HIGHLIGHT_EDGE if highlighted else _EDGE_COLOR
        linewidth = 3.0 if highlighted else 1.4
        linestyle = ":" if edge.style is EdgeStyle.DOTTED else "-"
        if edge.directed and arrow_style == "arrow":
            arrow = FancyArrowPatch(
                first,
                second,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
                shrinkA=11,
                shrinkB=11,
                zorder=2,
            )
            arrow.set_gid("dynkinlab-bigraph-arrow")
            ax.add_patch(arrow)
        else:
            ax.plot(
                (first[0], second[0]),
                (first[1], second[1]),
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                solid_capstyle="round",
                zorder=2,
            )
        if not show_multiplicity or edge.multiplicity <= 1:
            continue
        delta_x = second[0] - first[0]
        delta_y = second[1] - first[1]
        length = (delta_x * delta_x + delta_y * delta_y) ** 0.5
        if length:
            offset_x = _MULTIPLICITY_OFFSET * delta_y / length
            offset_y = _MULTIPLICITY_OFFSET * delta_x / length
        else:
            offset_x = offset_y = 0.0
        ax.text(
            (first[0] + second[0]) / 2 - offset_x,
            (first[1] + second[1]) / 2 + offset_y,
            f"x{edge.multiplicity}",
            color=color,
            fontsize=8,
            ha="center",
            va="center",
            zorder=4,
        )


class BigraphRenderer:
    """Draw a mixed bigraph on matplotlib axes or as inline SVG."""

    @classmethod
    def draw(
        cls,
        bigraph: Bigraph,
        *,
        ax: Axes | None = None,
        layout: str | None = None,
        seed: int | None = None,
        iterations: int | None = None,
        positions: Mapping[Hashable, tuple[float, float]] | None = None,
        options: DisplayOptions | None = None,
        title: str | None = None,
        highlight_edges: Iterable[frozenset[Hashable]] = (),
        highlight_nodes: Iterable[Hashable] = (),
        display_labels: Mapping[Hashable, str] | None = None,
    ) -> Axes:
        """Draw a faithful bigraph on an axes.

        Parameters
        ----------
        bigraph : Bigraph
            The bigraph to draw.
        ax : matplotlib.axes.Axes | None
            Existing axes. A new figure is created when omitted.
        layout : str | None
            Layout policy. The effective display options choose it when omitted.
        seed : int | None
            Layout seed. The effective display options choose it when omitted.
        iterations : int | None
            Spring iteration count. The effective display options choose it when
            omitted.
        positions : Mapping[Hashable, tuple[float, float]] | None
            Computed coordinates to reuse.
        options : DisplayOptions | None
            Resolved display options. The canonical options are used when omitted.
        title : str | None
            Optional axes title.
        highlight_edges : Iterable[frozenset[Hashable]]
            Undirected endpoint pairs to draw with the highlight style.
        highlight_nodes : Iterable[Hashable]
            Vertex labels to draw with the highlight style.
        display_labels : Mapping[Hashable, str] | None
            Optional text replacing the raw vertex labels.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the drawing.

        Raises
        ------
        ValueError
            If a position is missing or the matrix has no faithful bigraph.
        """
        settings = display.options if options is None else options
        if ax is None:
            figure = Figure(figsize=settings.bigraph_figure_size, dpi=100)
            ax = figure.add_subplot(111)
        if positions is None:
            selected_layout = settings.bigraph_layout if layout is None else layout
            selected_seed = settings.bigraph_layout_seed if seed is None else seed
            selected_iterations = (
                settings.bigraph_layout_iterations if iterations is None else iterations
            )
            positions = compute_layout(
                bigraph,
                selected_layout,
                selected_seed,
                selected_iterations,
            )
        missing = [label for label in bigraph.labels if label not in positions]
        if missing:
            raise ValueError(f"missing positions for labels: {missing!r}")
        points = {label: positions[label] for label in bigraph.labels}

        highlighted_edges = {frozenset(pair) for pair in highlight_edges}
        highlighted_nodes = set(highlight_nodes)
        labels = {} if display_labels is None else dict(display_labels)

        _draw_bigraph_edges(
            ax,
            bigraph,
            points,
            highlighted_edges,
            show_multiplicity=settings.bigraph_show_multiplicity,
            arrow_style=settings.bigraph_arrow_style,
        )

        node_colors = [
            _HIGHLIGHT_NODE if label in highlighted_nodes else _NODE_COLOR
            for label in bigraph.labels
        ]
        nodes = ax.scatter(
            [points[label][0] for label in bigraph.labels],
            [points[label][1] for label in bigraph.labels],
            s=_NODE_SIZE,
            c=node_colors,
            edgecolors="#2c3e50",
            linewidths=1.2,
            zorder=3,
        )
        nodes.set_gid("dynkinlab-bigraph-vertices")
        if settings.bigraph_show_labels:
            for label in bigraph.labels:
                x_coordinate, y_coordinate = points[label]
                ax.text(
                    x_coordinate,
                    y_coordinate,
                    labels.get(label, str(label)),
                    ha="center",
                    va="center",
                    fontsize=9,
                    zorder=4,
                )
        if title is not None:
            ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.12)
        ax.set_axis_off()
        return ax

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
        figure = Figure(figsize=options.bigraph_figure_size, dpi=100)
        FigureCanvasSVG(figure)
        axes = figure.add_subplot(111)
        cls.draw(bigraph, ax=axes, options=options)

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


def _centroid_positions(
    tree: BlockCutTree,
    positions: Mapping[Hashable, tuple[float, float]],
) -> dict[Hashable, tuple[float, float]]:
    """Place blocks at centroids and keep articulation points in place."""
    missing = [label for label in tree.labels if label not in positions]
    if missing:
        raise ValueError(f"missing positions for labels: {missing!r}")
    points: dict[Hashable, tuple[float, float]] = {
        label: positions[label] for label in tree.articulation_points
    }
    if not tree.blocks:
        for label in tree.labels:
            points[label] = positions[label]
        return points
    for block in tree.blocks:
        x_coordinate = sum(positions[label][0] for label in block.labels) / len(block.labels)
        y_coordinate = sum(positions[label][1] for label in block.labels) / len(block.labels)
        points[block] = (x_coordinate, y_coordinate)
    return points


class BlockCutTreeRenderer:
    """Draw a block-cut tree with a NetworkX bipartite layout."""

    @classmethod
    def draw(
        cls,
        tree: BlockCutTree,
        *,
        ax: Axes | None = None,
        options: DisplayOptions | None = None,
        title: str | None = None,
        positions: Mapping[Hashable, tuple[float, float]] | None = None,
    ) -> Axes:
        """Draw a block-cut tree on an axes.

        Parameters
        ----------
        tree : BlockCutTree
            The block-cut tree to draw.
        ax : matplotlib.axes.Axes | None
            Existing axes. A new figure is created when omitted.
        options : DisplayOptions | None
            Resolved display options. The canonical options are used when omitted.
        title : str | None
            Optional axes title.
        positions : Mapping[Hashable, tuple[float, float]] | None
            Coordinates from the original graph. When supplied, articulation
            points keep their positions and each block is placed at the
            centroid of its vertices.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the drawing.
        """
        settings = display.options if options is None else options
        if ax is None:
            figure = Figure(figsize=settings.bigraph_figure_size, dpi=100)
            ax = figure.add_subplot(111)

        graph = tree.to_networkx()
        points: dict[Hashable, tuple[float, float]]
        if positions is None:
            visible_labels = tree.labels
            if not tree.blocks:
                points = {tree.labels[0]: _SINGLE_VERTEX_POSITION}
            else:
                raw = nx.bipartite_layout(graph, nodes=tree.blocks, align="horizontal")
                block_positions = sorted(
                    (raw[block] for block in tree.blocks),
                    key=lambda position: (float(position[0]), float(position[1])),
                )
                vertex_positions = sorted(
                    (raw[label] for label in tree.labels),
                    key=lambda position: (float(position[0]), float(position[1])),
                )
                points = {
                    block: (float(position[0]), float(position[1]))
                    for block, position in zip(tree.blocks, block_positions, strict=True)
                }
                points.update(
                    {
                        label: (float(position[0]), float(position[1]))
                        for label, position in zip(
                            tree.labels,
                            vertex_positions,
                            strict=True,
                        )
                    },
                )
        else:
            points = _centroid_positions(tree, positions)
            visible_labels = tree.labels if not tree.blocks else tree.articulation_points

        for first, second in graph.edges:
            if first not in points or second not in points:
                continue
            ax.plot(
                (points[first][0], points[second][0]),
                (points[first][1], points[second][1]),
                color=_TREE_EDGE,
                linewidth=1.1,
                zorder=1,
            )

        articulation = set(tree.articulation_points)
        vertex_colors = [
            _HIGHLIGHT_NODE if label in articulation else _NODE_COLOR for label in visible_labels
        ]
        vertices = ax.scatter(
            [points[label][0] for label in visible_labels],
            [points[label][1] for label in visible_labels],
            s=420,
            c=vertex_colors,
            edgecolors="#2c3e50",
            linewidths=1.1,
            zorder=3,
        )
        vertices.set_gid("dynkinlab-tree-vertices")
        blocks = ax.scatter(
            [points[block][0] for block in tree.blocks],
            [points[block][1] for block in tree.blocks],
            s=_BLOCK_SIZE,
            c=_BLOCK_COLOR,
            marker="s",
            edgecolors="#2c3e50",
            linewidths=1.1,
            zorder=3,
        )
        blocks.set_gid("dynkinlab-tree-blocks")

        for label in visible_labels:
            x_coordinate, y_coordinate = points[label]
            ax.text(
                x_coordinate,
                y_coordinate,
                str(label),
                ha="center",
                va="center",
                fontsize=9,
                zorder=4,
            )
        for index, block in enumerate(tree.blocks, start=1):
            x_coordinate, y_coordinate = points[block]
            block_label = "{" + ", ".join(str(label) for label in block.labels) + "}"
            ax.text(
                x_coordinate,
                y_coordinate,
                f"B{index}",
                ha="center",
                va="center",
                fontsize=8,
                zorder=4,
            )
            ax.text(
                x_coordinate,
                y_coordinate + _BLOCK_LABEL_OFFSET,
                block_label,
                ha="center",
                va="bottom",
                fontsize=7,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none"},
                zorder=4,
            )
        if title is not None:
            ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.25)
        ax.set_axis_off()
        return ax
