# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for display options and notebook representations."""

import pytest

import dynkinlab
from dynkinlab import Bigraph, DynkinType, FlationSequence, FlationStep, Inertia, QuasiCartanMatrix
from dynkinlab.display import DisplayOptions, options


class TestDisplayOptions:
    """Verify defaults, mutation, validation, and reset."""

    def test_field_defaults(self) -> None:
        display_options = DisplayOptions()

        assert display_options.expanded_form_dimension_limit == 3
        assert display_options.matrix_index_base == 0
        assert display_options.bigraph_layout == "auto"
        assert display_options.bigraph_layout_iterations == 10000
        assert display_options.bigraph_layout_seed == 0
        assert display_options.bigraph_figure_size == (4.0, 4.0)
        assert display_options.bigraph_show_labels is True
        assert display_options.bigraph_show_multiplicity is True
        assert display_options.bigraph_arrow_style == "arrow"

    def test_canonical_instance_is_a_display_options_object(self) -> None:
        assert isinstance(options, DisplayOptions)

    def test_package_exposes_the_canonical_options(self) -> None:
        assert dynkinlab.display.options is options

    def test_valid_mutation(self) -> None:
        display_options = DisplayOptions()

        display_options.expanded_form_dimension_limit = 5
        display_options.matrix_index_base = 1
        display_options.bigraph_layout = "kamada_kawai"
        display_options.bigraph_layout_iterations = 500
        display_options.bigraph_layout_seed = 7
        display_options.bigraph_figure_size = (5, 3)
        display_options.bigraph_show_labels = False
        display_options.bigraph_show_multiplicity = False
        display_options.bigraph_arrow_style = "line"

        assert display_options.expanded_form_dimension_limit == 5
        assert display_options.matrix_index_base == 1
        assert display_options.bigraph_layout == "kamada_kawai"
        assert display_options.bigraph_layout_iterations == 500
        assert display_options.bigraph_layout_seed == 7
        assert display_options.bigraph_figure_size == (5.0, 3.0)
        assert not display_options.bigraph_show_labels
        assert not display_options.bigraph_show_multiplicity
        assert display_options.bigraph_arrow_style == "line"

    @pytest.mark.parametrize("field", ["expanded_form_dimension_limit", "matrix_index_base"])
    @pytest.mark.parametrize("value", [-1, 1.5, "3", True])
    def test_invalid_values_raise_value_error(self, field: str, value: object) -> None:
        display_options = DisplayOptions()

        with pytest.raises(ValueError, match=field):
            setattr(display_options, field, value)

    def test_reset_restores_defaults(self) -> None:
        display_options = DisplayOptions(
            expanded_form_dimension_limit=5,
            matrix_index_base=1,
            bigraph_layout="circular",
            bigraph_layout_iterations=500,
            bigraph_layout_seed=7,
            bigraph_figure_size=(5, 3),
            bigraph_show_labels=False,
            bigraph_show_multiplicity=False,
            bigraph_arrow_style="line",
        )

        display_options.reset()

        assert display_options.expanded_form_dimension_limit == 3
        assert display_options.matrix_index_base == 0
        assert display_options.bigraph_layout == "auto"
        assert display_options.bigraph_layout_iterations == 10000
        assert display_options.bigraph_layout_seed == 0
        assert display_options.bigraph_figure_size == (4.0, 4.0)
        assert display_options.bigraph_show_labels is True
        assert display_options.bigraph_show_multiplicity is True
        assert display_options.bigraph_arrow_style == "arrow"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("bigraph_layout_seed", -1),
            ("bigraph_layout_seed", True),
            ("bigraph_layout", "spiral"),
            ("bigraph_layout", None),
            ("bigraph_layout_iterations", 0),
            ("bigraph_layout_iterations", -1),
            ("bigraph_layout_iterations", True),
            ("bigraph_layout_iterations", 1.5),
            ("bigraph_arrow_style", ["arrow"]),
            ("bigraph_figure_size", (0, 4)),
            ("bigraph_figure_size", (4,)),
            ("bigraph_figure_size", ("4", 4)),
            ("bigraph_show_labels", 1),
            ("bigraph_show_multiplicity", "yes"),
            ("bigraph_arrow_style", "dashed"),
        ],
    )
    def test_invalid_bigraph_values_raise_value_error(
        self,
        field: str,
        value: object,
    ) -> None:
        display_options = DisplayOptions()

        with pytest.raises(ValueError, match=field):
            setattr(display_options, field, value)


class TestMatrixDisplay:
    """Verify Markdown and HTML matrix rendering."""

    def test_zero_based_markdown_table(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))

        markdown = matrix.to_markdown(options=DisplayOptions(matrix_index_base=0))

        assert "|   | 0 | 1 |" in markdown
        assert "| 0 | 2 | -1 |" in markdown
        assert "| 1 | -1 | 2 |" in markdown

    def test_one_based_markdown_table(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))

        markdown = matrix.to_markdown(options=DisplayOptions(matrix_index_base=1))

        assert "|   | 1 | 2 |" in markdown
        assert "| 1 | 2 | -1 |" in markdown
        assert "| 2 | -1 | 2 |" in markdown

    def test_html_table(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))

        html = matrix.to_html(options=DisplayOptions(matrix_index_base=0))

        assert '<table class="dynkinlab-matrix">' in html
        assert "<thead><tr><th></th><th>0</th><th>1</th></tr></thead>" in html
        assert "<tbody>" in html
        assert "<tr><th>0</th><td>2</td><td>-1</td></tr>" in html
        assert "<tr><th>1</th><td>-1</td><td>2</td></tr>" in html

    def test_repr_html_is_registered_and_uses_canonical_options(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))
        monkeypatch.setattr(dynkinlab.display.options, "matrix_index_base", 1)

        assert hasattr(matrix, "_repr_html_")
        html = matrix.to_html()

        assert "<th>1</th>" in html
        assert "<th>2</th>" in html


class TestLabeledMatrixDisplay:
    """Verify display-only labels and collapsed HTML output."""

    def test_html_table_uses_labels(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))

        html = matrix.to_html(labels=("a", "b"))

        assert "<th>a</th><th>b</th>" in html
        assert "<tr><th>a</th><td>2</td><td>-1</td></tr>" in html
        assert "<tr><th>b</th><td>-1</td><td>2</td></tr>" in html

    def test_markdown_table_uses_labels(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))

        markdown = matrix.to_markdown(labels=("a", "b"))

        assert "|   | a | b |" in markdown
        assert "| a | 2 | -1 |" in markdown
        assert "| b | -1 | 2 |" in markdown

    def test_non_contiguous_labels_are_preserved(self) -> None:
        matrix = DynkinType("D", 8).cartan_matrix()
        labels = (1, 2, 3, 4, 5, 7, 8, 9)

        markdown = matrix.to_markdown(labels=labels)

        assert "|   | 1 | 2 | 3 | 4 | 5 | 7 | 8 | 9 |" in markdown

    @pytest.mark.parametrize(
        "labels",
        [("a",), ("a", "a"), ("a", ["b"])],
    )
    def test_invalid_labels_are_rejected(self, labels: tuple[object, ...]) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))

        with pytest.raises(ValueError, match="labels"):
            matrix.to_markdown(labels=labels)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="labels"):
            matrix.to_html(labels=labels)  # type: ignore[arg-type]

    def test_labels_are_escaped(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))

        markdown = matrix.to_markdown(labels=("a|b", "<c>"))
        html = matrix.to_html(labels=("a|b", "<c>"))

        assert r"a\|b" in markdown
        assert "&lt;c&gt;" in html

    def test_collapsed_html_names_the_dimension(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))

        html = matrix.to_html(collapse=True)

        assert "<details" in html
        assert "<summary>Matrix (2 x 2)</summary>" in html
        assert '<table class="dynkinlab-matrix">' in html

    def test_default_html_stays_expanded(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))

        assert "<details" not in matrix.to_html()
        assert "<details" not in matrix._repr_html_()


class TestQuadraticFormDisplay:
    """Verify Markdown formula rendering."""

    def test_small_form_is_expanded(self) -> None:
        form = QuasiCartanMatrix(((2, -1), (-1, 2))).quadratic_form

        markdown = form.to_markdown(
            options=DisplayOptions(expanded_form_dimension_limit=3),
        )

        assert "**Quadratic form**" in markdown
        assert "x_0^2 + x_1^2 - x_0 x_1" in markdown

    def test_large_form_stays_general(self) -> None:
        form = QuasiCartanMatrix(
            (
                (2, 0, 0, 0),
                (0, 2, 0, 0),
                (0, 0, 2, 0),
                (0, 0, 0, 2),
            ),
        ).quadratic_form

        markdown = form.to_markdown(
            options=DisplayOptions(expanded_form_dimension_limit=3),
        )

        assert r"\sum_i" in markdown
        assert "x_0^2 + x_1^2 + x_2^2 + x_3^2" not in markdown

    def test_zero_coefficients_are_omitted(self) -> None:
        form = QuasiCartanMatrix(((2, 0), (0, 2))).quadratic_form

        markdown = form.to_markdown(
            options=DisplayOptions(expanded_form_dimension_limit=3),
        )

        assert "x_0^2 + x_1^2" in markdown
        assert "x_0 x_1" not in markdown

    def test_negative_coefficients_keep_their_magnitude(self) -> None:
        form = QuasiCartanMatrix(((2, -2), (-2, 2))).quadratic_form

        markdown = form.to_markdown(
            options=DisplayOptions(expanded_form_dimension_limit=3),
        )

        assert "x_0^2 + x_1^2 - 2 x_0 x_1" in markdown

    def test_unit_coefficient_has_no_numeric_factor(self) -> None:
        form = QuasiCartanMatrix(((2, 1), (1, 2))).quadratic_form

        markdown = form.to_markdown(
            options=DisplayOptions(expanded_form_dimension_limit=3),
        )

        assert "x_0^2 + x_1^2 + x_0 x_1" in markdown
        assert "1 x_0 x_1" not in markdown

    def test_zero_limit_uses_only_the_general_formula(self) -> None:
        form = QuasiCartanMatrix(((2, -1), (-1, 2))).quadratic_form

        markdown = form.to_markdown(
            options=DisplayOptions(expanded_form_dimension_limit=0),
        )

        assert r"\sum_i" in markdown
        assert "x_0^2 + x_1^2 - x_0 x_1" not in markdown

    def test_repr_markdown_is_registered_and_uses_canonical_options(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        form = QuasiCartanMatrix(((2, -1), (-1, 2))).quadratic_form
        monkeypatch.setattr(dynkinlab.display.options, "expanded_form_dimension_limit", 0)

        assert hasattr(form, "_repr_markdown_")
        markdown = form.to_markdown()

        assert "x_0^2 + x_1^2 - x_0 x_1" not in markdown


class TestInertiaDisplay:
    """Verify Markdown inertia summaries."""

    def test_positive_definite_inertia(self) -> None:
        markdown = Inertia(2, 0, 0).to_markdown()

        assert "| 2 | 0 | 0 | 2 | 2 |" in markdown
        assert "positive definite" in markdown

    def test_negative_definite_inertia(self) -> None:
        markdown = Inertia(0, 2, 0).to_markdown()

        assert "negative definite" in markdown

    def test_degenerate_inertia(self) -> None:
        markdown = Inertia(2, 0, 1).to_markdown()

        assert "| 2 | 0 | 1 | 2 | 2 |" in markdown
        assert "degenerate" in markdown

    def test_indefinite_inertia(self) -> None:
        markdown = Inertia(1, 1, 0).to_markdown()

        assert "| 1 | 1 | 0 | 2 | 0 |" in markdown
        assert "indefinite" in markdown

    def test_repr_markdown_is_registered(self) -> None:
        inertia = Inertia(2, 0, 0)

        assert hasattr(inertia, "_repr_markdown_")
        assert "positive definite" in inertia.to_markdown()


class TestDisplayPrecedenceAndRepr:
    """Verify option precedence, reset behavior, and unchanged debug output."""

    def test_per_call_matrix_options_override_canonical(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))
        monkeypatch.setattr(dynkinlab.display.options, "matrix_index_base", 1)

        markdown = matrix.to_markdown(options=DisplayOptions(matrix_index_base=0))

        assert "|   | 0 | 1 |" in markdown
        assert dynkinlab.display.options.matrix_index_base == 1

    def test_per_call_form_options_override_canonical(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        form = QuasiCartanMatrix(((2, -1), (-1, 2))).quadratic_form
        monkeypatch.setattr(dynkinlab.display.options, "expanded_form_dimension_limit", 0)

        markdown = form.to_markdown(
            options=DisplayOptions(expanded_form_dimension_limit=3),
        )

        assert "x_0^2 + x_1^2 - x_0 x_1" in markdown
        assert dynkinlab.display.options.expanded_form_dimension_limit == 0

    def test_reset_restores_default_rendering(self, monkeypatch: pytest.MonkeyPatch) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))
        form = matrix.quadratic_form
        monkeypatch.setattr(dynkinlab.display.options, "matrix_index_base", 1)
        monkeypatch.setattr(dynkinlab.display.options, "expanded_form_dimension_limit", 0)

        dynkinlab.display.options.reset()

        assert dynkinlab.display.options.matrix_index_base == 0
        assert dynkinlab.display.options.expanded_form_dimension_limit == 3
        assert "|   | 0 | 1 |" in matrix.to_markdown()
        assert "x_0^2 + x_1^2 - x_0 x_1" in form.to_markdown()

    def test_repr_remains_dataclass_style(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))
        form = matrix.quadratic_form
        inertia = matrix.inertia

        expected_prefixes = (
            (matrix, "QuasiCartanMatrix("),
            (form, "QuadraticForm("),
            (inertia, "Inertia("),
        )
        for value, prefix in expected_prefixes:
            text = repr(value)
            assert text.startswith(prefix)
            assert "<table" not in text
            assert "|" not in text


class TestNonSymmetricMatrixDisplay:
    """Verify the general matrix still displays its original entries."""

    def test_markdown_uses_original_entries(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-2, 2)))

        markdown = matrix.to_markdown(options=DisplayOptions(matrix_index_base=0))

        assert "|   | 0 | 1 |" in markdown
        assert "| 0 | 2 | -1 |" in markdown
        assert "| 1 | -2 | 2 |" in markdown

    def test_html_and_hook_use_original_entries(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-2, 2)))

        html = matrix.to_html(options=DisplayOptions(matrix_index_base=0))

        assert "<td>-2</td>" in html
        assert matrix._repr_html_() == matrix.to_html()


class TestBigraphDisplay:
    """Verify bigraph Markdown, HTML, and display options."""

    def test_markdown_summarizes_arrow(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, -1), (-2, 2))), ("a", "b"))

        markdown = bigraph.to_markdown()

        assert "| a | b | arrow | solid | 2 |" in markdown

    def test_markdown_distinguishes_dotted_edges(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, 1), (1, 2))), ("a", "b"))

        markdown = bigraph.to_markdown()

        assert "| a | b | edge | dotted | 1 |" in markdown

    def test_markdown_multiplicity_can_be_hidden(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, -1), (-2, 2))), ("a", "b"))

        hidden = bigraph.to_markdown(
            options=DisplayOptions(bigraph_show_multiplicity=False),
        )

        assert "multiplicity" not in hidden
        assert "multiplicity" in bigraph.to_markdown()

    def test_canonical_multiplicity_option_and_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, -1), (-2, 2))), ("a", "b"))
        monkeypatch.setattr(dynkinlab.display.options, "bigraph_show_multiplicity", False)

        assert "multiplicity" not in bigraph.to_markdown()
        assert "multiplicity" in bigraph.to_markdown(
            options=DisplayOptions(bigraph_show_multiplicity=True),
        )
        assert dynkinlab.display.options.bigraph_show_multiplicity is False
        dynkinlab.display.options.reset()

    def test_html_contains_svg_and_edge_summary(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, -1), (-2, 2))), ("a", "b"))

        html = bigraph.to_html()

        assert "<svg" in html
        assert 'data-edges="a -&gt; b (solid 2)"' in html
        assert "a" in html and "b" in html
        assert bigraph._repr_html_() == bigraph.to_html()

    def test_dotted_html_uses_dashed_stroke(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, 1), (1, 2))), ("a", "b"))

        html = bigraph.to_html()

        assert "stroke-dasharray" in html


class TestDynkinDisplay:
    """Verify Dynkin type and classification rendering."""

    def test_type_markdown_and_hook(self) -> None:
        dynkin_type = DynkinType("A", 3)

        assert "A_3" in dynkin_type.to_markdown()
        assert dynkin_type._repr_markdown_() == dynkin_type.to_markdown()

    def test_classification_markdown_and_hook(self) -> None:
        result = QuasiCartanMatrix(((2, -1), (-1, 2))).classify()

        markdown = result.to_markdown()

        assert "CLASSIFIED" in markdown
        assert "`A_2`" in markdown
        assert "Steps:" in markdown
        assert "positive definite" in markdown
        assert result._repr_markdown_() == markdown

    def test_non_finite_classification_renders_diagnostics(self) -> None:
        result = QuasiCartanMatrix(
            ((2, -1, -1), (-1, 2, -1), (-1, -1, 2)),
        ).classify()

        markdown = result.to_markdown()

        assert "NOT_POSITIVE_DEFINITE" in markdown
        assert "degenerate" in markdown


class TestNewValueRepr:
    """Verify new value objects keep plain dataclass-style repr."""

    def test_repr_stays_plain(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, -1), (-2, 2))), ("a", "b"))
        result = bigraph.classify()
        sequence = FlationSequence((FlationStep("a", "b"),))

        values_and_prefixes = (
            (bigraph, "Bigraph("),
            (DynkinType("A", 3), "DynkinType("),
            (result, "DynkinClassification("),
            (sequence, "FlationSequence("),
        )
        for value, prefix in values_and_prefixes:
            text = repr(value)
            assert text.startswith(prefix)
            assert "<table" not in text
            assert "|" not in text
