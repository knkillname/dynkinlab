# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for NetworkX layouts, bigraph drawing, and block-cut drawing."""

from collections.abc import Hashable, Mapping

import networkx as nx
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch

from dynkinlab import Bigraph, DynkinType, QuasiCartanMatrix

_A9_TEXT = """1--2..3--1
3..4--5--3
3--9--4
5..9
5..6..8..5
4--7"""


def _a9() -> Bigraph:
    return Bigraph.from_paths(_A9_TEXT, labels=tuple(range(1, 10)))


def _mixed() -> Bigraph:
    matrix = QuasiCartanMatrix(((2, -1, 1), (-2, 2, 0), (1, 0, 2)))
    return Bigraph(matrix, ("a", "b", "c"))


class TestLayout:
    """Verify the NetworkX layout policy and repeatability."""

    def test_auto_prefers_kamada_kawai_for_connected_graphs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bigraph = DynkinType("A", 5).bigraph()
        expected: dict[Hashable, tuple[float, float]] = {
            label: (float(index), 0.0) for index, label in enumerate(bigraph.labels)
        }

        def fake_kamada(
            graph: "nx.Graph[Hashable]",
            weight: str | None = "weight",
        ) -> dict[Hashable, tuple[float, float]]:
            assert weight is None
            return expected

        monkeypatch.setattr("dynkinlab.bigraph_display.nx.kamada_kawai_layout", fake_kamada)

        assert bigraph.layout() == expected

    def test_auto_relaxes_planar_k4_graphs(self) -> None:
        bigraph = _a9()
        graph = bigraph.to_networkx()
        planar = nx.planar_layout(graph)
        expected = nx.spring_layout(
            graph,
            pos=planar,
            iterations=10000,
            seed=7,
            weight=None,
            method="energy",
        )

        assert bigraph.layout(seed=7) == {
            label: (float(position[0]), float(position[1])) for label, position in expected.items()
        }

    def test_auto_falls_back_to_spring_without_scipy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bigraph = DynkinType("A", 5).bigraph()
        expected: dict[Hashable, tuple[float, float]] = {
            label: (float(index), 0.0) for index, label in enumerate(bigraph.labels)
        }

        def fake_kamada(
            graph: "nx.Graph[Hashable]",
            weight: str | None = "weight",
        ) -> dict[Hashable, tuple[float, float]]:
            raise ModuleNotFoundError("scipy")

        def fake_spring(
            graph: "nx.Graph[Hashable]",
            pos: Mapping[Hashable, tuple[float, float]] | None = None,
            seed: int | None = None,
            iterations: int = 50,
            weight: str | None = "weight",
            method: str = "auto",
        ) -> dict[Hashable, tuple[float, float]]:
            assert pos is None
            assert seed == 7
            assert iterations == 500
            assert weight is None
            assert method == "energy"
            return expected

        monkeypatch.setattr("dynkinlab.bigraph_display.nx.kamada_kawai_layout", fake_kamada)
        monkeypatch.setattr("dynkinlab.bigraph_display.nx.spring_layout", fake_spring)

        assert bigraph.layout(seed=7, iterations=500) == expected

    def test_auto_uses_spring_for_disconnected_graphs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bigraph = Bigraph.from_paths("a--b\nc--d")
        expected: dict[Hashable, tuple[float, float]] = {
            "a": (0.0, 0.0),
            "b": (1.0, 0.0),
            "c": (2.0, 0.0),
            "d": (3.0, 0.0),
        }

        def fake_kamada(
            graph: "nx.Graph[Hashable]",
            weight: str | None = "weight",
        ) -> dict[Hashable, tuple[float, float]]:
            raise AssertionError("Kamada-Kawai requires a connected graph")

        def fake_spring(
            graph: "nx.Graph[Hashable]",
            pos: Mapping[Hashable, tuple[float, float]] | None = None,
            seed: int | None = None,
            iterations: int = 50,
            weight: str | None = "weight",
            method: str = "auto",
        ) -> dict[Hashable, tuple[float, float]]:
            assert pos is None
            assert seed == 7
            assert iterations == 10000
            assert weight is None
            assert method == "energy"
            return expected

        monkeypatch.setattr("dynkinlab.bigraph_display.nx.kamada_kawai_layout", fake_kamada)
        monkeypatch.setattr("dynkinlab.bigraph_display.nx.spring_layout", fake_spring)

        assert bigraph.layout(seed=7) == expected

    @pytest.mark.parametrize("layout", ["planar", "spring", "circular"])
    def test_named_layouts_are_available(self, layout: str) -> None:
        bigraph = DynkinType("A", 5).bigraph()

        if layout == "planar":
            expected = nx.planar_layout(bigraph.to_networkx())
        elif layout == "spring":
            expected = nx.spring_layout(
                bigraph.to_networkx(),
                seed=11,
                iterations=10000,
                weight=None,
                method="energy",
            )
        else:
            expected = nx.circular_layout(bigraph.to_networkx())

        expected_points = {
            label: (float(position[0]), float(position[1])) for label, position in expected.items()
        }
        assert bigraph.layout(layout, seed=11) == expected_points

    def test_spring_falls_back_to_force_without_scipy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bigraph = DynkinType("A", 5).bigraph()
        expected: dict[Hashable, tuple[float, float]] = {
            label: (float(index), 0.0) for index, label in enumerate(bigraph.labels)
        }
        calls: list[str] = []

        def fake_spring(
            graph: "nx.Graph[Hashable]",
            pos: Mapping[Hashable, tuple[float, float]] | None = None,
            seed: int | None = None,
            iterations: int = 50,
            weight: str | None = "weight",
            method: str = "auto",
        ) -> dict[Hashable, tuple[float, float]]:
            calls.append(method)
            assert weight is None
            if method == "energy":
                raise ModuleNotFoundError("scipy")
            assert method == "force"
            assert iterations == 200
            return expected

        monkeypatch.setattr("dynkinlab.bigraph_display.nx.spring_layout", fake_spring)

        assert bigraph.layout("spring", iterations=5000) == expected
        assert calls == ["energy", "force"]

    def test_spring_uses_configured_iterations(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bigraph = _a9()
        expected: dict[Hashable, tuple[float, float]] = {
            label: (0.0, float(index)) for index, label in enumerate(bigraph.labels)
        }

        def fake_spring(
            graph: "nx.Graph[Hashable]",
            pos: Mapping[Hashable, tuple[float, float]] | None = None,
            seed: int | None = None,
            iterations: int = 50,
            weight: str | None = "weight",
            method: str = "auto",
        ) -> dict[Hashable, tuple[float, float]]:
            assert pos is None
            assert seed == 3
            assert iterations == 321
            assert weight is None
            assert method == "energy"
            return expected

        monkeypatch.setattr("dynkinlab.bigraph_display.nx.spring_layout", fake_spring)

        assert bigraph.layout("spring", seed=3, iterations=321) == expected

    def test_kamada_kawai_delegates_to_networkx(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bigraph = DynkinType("A", 5).bigraph()
        expected: dict[Hashable, tuple[float, float]] = {
            label: (float(label), 0.0) for label in range(5)
        }

        def fake_layout(
            graph: "nx.Graph[Hashable]",
            weight: str | None = "weight",
        ) -> dict[Hashable, tuple[float, float]]:
            assert weight is None
            return expected

        monkeypatch.setattr("dynkinlab.bigraph_display.nx.kamada_kawai_layout", fake_layout)

        assert bigraph.layout("kamada_kawai") == expected

    def test_unknown_layout_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown bigraph layout"):
            _a9().layout("spiral")

    def test_seeded_spring_layout_is_repeatable(self) -> None:
        bigraph = _a9()

        assert bigraph.layout("spring", seed=13) == bigraph.layout("spring", seed=13)

    def test_single_vertex_is_placed_at_the_origin(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2,),)), ("solo",))

        assert bigraph.layout() == {"solo": (0.0, 0.0)}


class TestBigraphDrawing:
    """Verify matplotlib rendering without IPython."""

    def test_mixed_styles_render_arrow_and_dotted_line(self) -> None:
        bigraph = _mixed()

        axes = bigraph.draw()

        arrows = [
            patch
            for patch in axes.patches
            if isinstance(patch, FancyArrowPatch) and patch.get_gid() == "dynkinlab-bigraph-arrow"
        ]
        dotted = [line for line in axes.lines if line.get_linestyle() == ":"]
        assert len(arrows) == 1
        assert len(dotted) == 1
        assert any(text.get_text() == "x2" for text in axes.texts)
        assert axes.axison is False
        assert axes.figure is not None

    def test_highlights_are_applied(self) -> None:
        bigraph = _mixed()

        axes = bigraph.draw(
            highlight_edges=(frozenset(("a", "c")),),
            highlight_nodes=("b",),
        )

        dotted = next(line for line in axes.lines if line.get_linestyle() == ":")
        assert dotted.get_color() == "#b03a2e"
        nodes = next(
            collection
            for collection in axes.collections
            if collection.get_gid() == "dynkinlab-bigraph-vertices"
        )
        colors = nodes.get_facecolor()
        assert np.array_equal(colors[1], np.asarray(to_rgba("#f7dc6f")))

    def test_supplied_positions_are_reused(self) -> None:
        bigraph = _mixed()
        positions: dict[Hashable, tuple[float, float]] = {
            "a": (0.0, 0.0),
            "b": (1.0, 0.0),
            "c": (0.0, 1.0),
        }

        axes = bigraph.draw(positions=positions)

        nodes = next(
            collection
            for collection in axes.collections
            if collection.get_gid() == "dynkinlab-bigraph-vertices"
        )
        offsets = np.asarray(nodes.get_offsets())
        assert [tuple(point) for point in offsets.tolist()] == [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
        ]

    def test_missing_positions_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing positions"):
            _mixed().draw(positions={"a": (0.0, 0.0)})

    def test_non_faithful_input_is_rejected(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, 2), (4, 2))), (1, 2))

        with pytest.raises(ValueError, match="no faithful bigraph"):
            bigraph.draw()

    def test_display_labels_replace_raw_labels(self) -> None:
        bigraph = _mixed()

        axes = bigraph.draw(display_labels={"a": "1=6"})

        assert "1=6" in [text.get_text() for text in axes.texts]


class TestBlockCutTreeDrawing:
    """Verify the bipartite block-cut tree figure."""

    def test_a9_tree_renders_all_nodes_and_block_labels(self) -> None:
        certificate = _a9().a_type_certificate()
        assert certificate is not None

        axes = certificate.components[0].tree.draw()

        texts = [text.get_text() for text in axes.texts]
        for label in range(1, 10):
            assert str(label) in texts
        for block_label in ("B1", "B2", "B3", "B4"):
            assert block_label in texts
        assert "{1, 2, 3}" in texts
        assert "{3, 4, 5, 9}" in texts
        vertices = next(
            collection
            for collection in axes.collections
            if collection.get_gid() == "dynkinlab-tree-vertices"
        )
        blocks = next(
            collection
            for collection in axes.collections
            if collection.get_gid() == "dynkinlab-tree-blocks"
        )
        assert len(np.asarray(vertices.get_offsets())) == 9
        assert len(np.asarray(blocks.get_offsets())) == 4

    def test_articulation_points_are_highlighted(self) -> None:
        certificate = _a9().a_type_certificate()
        assert certificate is not None

        axes = certificate.components[0].tree.draw()

        vertices = next(
            collection
            for collection in axes.collections
            if collection.get_gid() == "dynkinlab-tree-vertices"
        )
        colors = vertices.get_facecolor()
        for index, label in enumerate(range(1, 10)):
            expected = to_rgba("#f7dc6f") if label in {3, 4, 5} else to_rgba("#ecf0f1")
            assert np.array_equal(colors[index], np.asarray(expected))

    def test_centroid_layout_keeps_cut_vertices_and_centers_blocks(self) -> None:
        bigraph = _a9()
        certificate = bigraph.a_type_certificate()
        assert certificate is not None
        tree = certificate.components[0].tree
        positions = bigraph.layout()

        axes = tree.draw(positions=positions)

        vertices = next(
            collection
            for collection in axes.collections
            if collection.get_gid() == "dynkinlab-tree-vertices"
        )
        vertex_points = [tuple(point) for point in np.asarray(vertices.get_offsets()).tolist()]
        expected_vertices = [positions[label] for label in tree.articulation_points]
        assert vertex_points == expected_vertices

        blocks = next(
            collection
            for collection in axes.collections
            if collection.get_gid() == "dynkinlab-tree-blocks"
        )
        block_points = [tuple(point) for point in np.asarray(blocks.get_offsets()).tolist()]
        expected_blocks = [
            (
                sum(positions[label][0] for label in block.labels) / len(block.labels),
                sum(positions[label][1] for label in block.labels) / len(block.labels),
            )
            for block in tree.blocks
        ]
        assert block_points == expected_blocks
        assert len(axes.lines) == 6
        vertex_texts = [text.get_text() for text in axes.texts if text.get_text().isdigit()]
        assert set(vertex_texts) == {"3", "4", "5"}

    def test_centroid_layout_requires_original_positions(self) -> None:
        certificate = _a9().a_type_certificate()
        assert certificate is not None
        tree = certificate.components[0].tree

        with pytest.raises(ValueError, match="missing positions"):
            tree.draw(positions={1: (0.0, 0.0)})

    def test_tree_layout_is_repeatable(self) -> None:
        certificate = _a9().a_type_certificate()
        assert certificate is not None
        tree = certificate.components[0].tree

        first = tree.draw()
        second = tree.draw()

        first_vertices = next(
            collection
            for collection in first.collections
            if collection.get_gid() == "dynkinlab-tree-vertices"
        )
        second_vertices = next(
            collection
            for collection in second.collections
            if collection.get_gid() == "dynkinlab-tree-vertices"
        )
        assert np.array_equal(first_vertices.get_offsets(), second_vertices.get_offsets())
        assert [text.get_position() for text in first.texts] == [
            text.get_position() for text in second.texts
        ]

    def test_single_vertex_tree_renders(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2,),)), ("solo",))

        axes = bigraph.block_cut_forest().trees[0].draw()

        vertices = next(
            collection
            for collection in axes.collections
            if collection.get_gid() == "dynkinlab-tree-vertices"
        )
        assert len(np.asarray(vertices.get_offsets())) == 1
