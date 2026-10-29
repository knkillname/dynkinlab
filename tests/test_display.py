# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for display options and notebook representations."""

import pytest

import dynkinlab
from dynkinlab import Inertia, QuasiCartanMatrix
from dynkinlab.display import DisplayOptions, options


class TestDisplayOptions:
    """Verify defaults, mutation, validation, and reset."""

    def test_field_defaults(self) -> None:
        display_options = DisplayOptions()

        assert display_options.expanded_form_dimension_limit == 3
        assert display_options.matrix_index_base == 0

    def test_canonical_instance_is_a_display_options_object(self) -> None:
        assert isinstance(options, DisplayOptions)

    def test_package_exposes_the_canonical_options(self) -> None:
        assert dynkinlab.display.options is options

    def test_valid_mutation(self) -> None:
        display_options = DisplayOptions()

        display_options.expanded_form_dimension_limit = 5
        display_options.matrix_index_base = 1

        assert display_options.expanded_form_dimension_limit == 5
        assert display_options.matrix_index_base == 1

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
        )

        display_options.reset()

        assert display_options.expanded_form_dimension_limit == 3
        assert display_options.matrix_index_base == 0


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
