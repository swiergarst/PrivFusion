"""Tests for utility functions and type classification."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pandas as pd

from privfusion.utils.type_classification import (
    ISO3,
    Continent,
    DateTime,
    Gender,
    ISOMixed,
    Numeric,
    extract_types,
    find_common_ancestors,
    find_mapping,
    get_type_mapping,
)
from privfusion.utils.utils import print_colored_dict, print_colored_list_of_dicts, print_colored_nested_dict


class TestPrintFunctions:
    """Tests for colored print utility functions."""

    def test_print_colored_dict_simple(self):
        """Test printing a simple dictionary."""
        test_dict = {"key1": "value1", "key2": "value2"}

        with patch("sys.stdout", new=StringIO()) as fake_out:
            print_colored_dict(test_dict)
            output = fake_out.getvalue()

            assert "key1" in output
            assert "value1" in output
            assert "key2" in output
            assert "value2" in output

    def test_print_colored_dict_nested(self):
        """Test printing a nested dictionary."""
        test_dict = {
            "outer_key": {
                "inner_key1": "inner_value1",
                "inner_key2": "inner_value2",
            },
        }

        with patch("sys.stdout", new=StringIO()) as fake_out:
            print_colored_dict(test_dict)
            output = fake_out.getvalue()

            assert "outer_key" in output
            assert "inner_key1" in output
            assert "inner_value1" in output

    def test_print_colored_dict_with_prefix(self):
        """Test printing dictionary with custom prefix."""
        test_dict = {"key": "value"}

        with patch("sys.stdout", new=StringIO()) as fake_out:
            print_colored_dict(test_dict, prefix="  ")
            output = fake_out.getvalue()

            assert "key" in output
            assert "value" in output

    def test_print_colored_list_of_dicts(self):
        """Test printing a list of dictionaries."""
        test_list = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]

        with patch("sys.stdout", new=StringIO()) as fake_out:
            print_colored_list_of_dicts(test_list)
            output = fake_out.getvalue()

            assert "Entry 1" in output
            assert "Entry 2" in output
            assert "Alice" in output
            assert "Bob" in output

    def test_print_colored_list_of_dicts_empty(self):
        """Test printing an empty list."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            print_colored_list_of_dicts([])
            output = fake_out.getvalue()

            assert output == ""

    def test_print_colored_nested_dict(self):
        """Test printing a nested dictionary structure."""
        test_dict = {
            "category1": [
                {"item": "value1"},
                {"item": "value2"},
            ],
            "category2": [
                {"item": "value3"},
            ],
        }

        with patch("sys.stdout", new=StringIO()) as fake_out:
            print_colored_nested_dict(test_dict)
            output = fake_out.getvalue()

            assert "category1" in output
            assert "category2" in output
            assert "value1" in output
            assert "value3" in output


class TestIdentifiers:
    """Tests for identifier classes."""

    def test_datetime_identifier(self):
        """Test DateTime identifier."""
        dt = DateTime()

        assert dt.is_of_this_type("2023-01-01")
        assert dt.is_of_this_type("2023-01-01 12:00:00")
        assert not dt.is_of_this_type("not a date")
        assert not dt.is_of_this_type("12345")

    def test_gender_identifier(self):
        """Test Gender identifier."""
        gender = Gender()

        assert gender.is_of_this_type("Male")
        assert gender.is_of_this_type("Female")
        assert gender.is_of_this_type("male")
        assert not gender.is_of_this_type("Unknown")
        assert not gender.is_of_this_type("123")

    def test_numeric_identifier(self):
        """Test Numeric identifier."""
        numeric = Numeric()

        assert numeric.is_of_this_type("123")
        assert numeric.is_of_this_type("123.456")
        assert numeric.is_of_this_type("-123.456")
        assert numeric.is_of_this_type("0")
        assert not numeric.is_of_this_type("abc")
        assert not numeric.is_of_this_type("12a34")

    def test_continent_identifier(self):
        """Test Continent identifier."""
        continent = Continent()

        assert continent.is_of_this_type("Africa")
        assert continent.is_of_this_type("Asia")
        assert continent.is_of_this_type("Europe")
        assert not continent.is_of_this_type("Atlantis")
        assert not continent.is_of_this_type("123")

    def test_iso3_identifier(self):
        """Test ISO3 identifier."""
        iso3 = ISO3()

        assert iso3.is_of_this_type("USA")
        assert iso3.is_of_this_type("GBR")
        assert iso3.is_of_this_type("FRA")
        assert not iso3.is_of_this_type("US")  # ISO2, not ISO3
        assert not iso3.is_of_this_type("USAA")  # Too long
        assert not iso3.is_of_this_type("123")

    def test_iso_mixed_identifier(self):
        """Test ISOMixed identifier."""
        iso_mixed = ISOMixed()

        assert iso_mixed.is_of_this_type("ID-AC")
        assert iso_mixed.is_of_this_type("ID-JK")
        assert iso_mixed.is_of_this_type("IDN")
        assert not iso_mixed.is_of_this_type("ID-XX")
        assert not iso_mixed.is_of_this_type("USA")


class TestTypeExtraction:
    """Tests for type extraction functions."""

    def test_extract_types_numeric(self):
        """Test extracting numeric types."""
        df = pd.DataFrame(
            {
                "numbers": [1, 2, 3, 4, 5],
                "floats": [1.1, 2.2, 3.3, 4.4, 5.5],
            },
        )

        types = extract_types(df)

        assert len(types) == 2
        assert types[0] == "Numeric"
        assert types[1] == "Numeric"

    def test_extract_types_datetime(self):
        """Test extracting datetime types."""
        df = pd.DataFrame(
            {
                "dates": pd.date_range("2023-01-01", periods=5),
            },
        )

        types = extract_types(df)

        assert len(types) == 1
        assert types[0] == "DateTime"

    def test_extract_types_mixed(self):
        """Test extracting mixed types."""
        df = pd.DataFrame(
            {
                "numbers": [1, 2, 3],
                "countries": ["USA", "GBR", "FRA"],
                "text": ["hello", "world", "test"],
            },
        )

        types = extract_types(df)

        assert len(types) == 3
        assert types[0] == "Numeric"
        assert types[1] == "ISOCode3"
        assert types[2] is None  # Generic text

    def test_extract_types_empty_dataframe(self):
        """Test extracting types from empty dataframe."""
        df = pd.DataFrame()

        types = extract_types(df)

        assert len(types) == 0

    def test_extract_types_with_nulls(self):
        """Test extracting types with null values."""
        df = pd.DataFrame(
            {
                "numbers": [1, 2, None, 4, 5],
                "mixed": [1, "text", None, 3, "more"],
            },
        )

        types = extract_types(df)

        assert len(types) == 2
        # Should still identify as numeric despite nulls
        assert types[0] == "Numeric"

    def test_get_type_mapping(self):
        """Test getting URI type mappings."""
        df = pd.DataFrame(
            {
                "numbers": [1, 2, 3],
                "countries": ["USA", "GBR", "FRA"],
            },
        )

        mappings = get_type_mapping(df)

        assert len(mappings) == 2
        assert mappings[0] == "http://www.w3.org/2001/XMLSchema#Number"
        assert mappings[1] == "https://dbpedia.org/page/ISO_3166-3"

    def test_get_type_mapping_with_none(self):
        """Test type mapping with unidentified types."""
        df = pd.DataFrame(
            {
                "unknown": ["random", "text", "values"],
            },
        )

        mappings = get_type_mapping(df)

        assert len(mappings) == 1
        assert mappings[0] is None


class TestSPARQLFunctions:
    """Tests for SPARQL-based functions."""

    @patch("privfusion.utils.type_classification.SPARQLWrapper")
    def test_find_mapping_success(self, mock_sparql_wrapper):
        """Test successful term mapping."""
        mock_sparql = MagicMock()
        mock_sparql_wrapper.return_value = mock_sparql

        mock_results = {
            "results": {
                "bindings": [
                    {"uri": {"value": "http://dbpedia.org/resource/Horse"}},
                    {"uri": {"value": "http://dbpedia.org/resource/Horse_(disambiguation)"}},
                ],
            },
        }
        mock_sparql.queryAndConvert.return_value = mock_results

        result = find_mapping("Horse")

        assert result is not None
        assert len(result) == 2
        assert "http://dbpedia.org/resource/Horse" in result

    @patch("privfusion.utils.type_classification.SPARQLWrapper")
    def test_find_mapping_no_results(self, mock_sparql_wrapper):
        """Test term mapping with no results."""
        mock_sparql = MagicMock()
        mock_sparql_wrapper.return_value = mock_sparql

        mock_results = {
            "results": {
                "bindings": [],
            },
        }
        mock_sparql.queryAndConvert.return_value = mock_results

        result = find_mapping("NonexistentTerm")

        assert result == []

    def test_find_mapping_none_input(self):
        """Test find_mapping with None input."""
        result = find_mapping(None)

        assert result is None

    @patch("privfusion.utils.type_classification.SPARQLWrapper")
    def test_find_mapping_error(self, mock_sparql_wrapper):
        """Test find_mapping with SPARQL error."""
        mock_sparql = MagicMock()
        mock_sparql_wrapper.return_value = mock_sparql
        mock_sparql.queryAndConvert.side_effect = Exception("SPARQL error")

        result = find_mapping("ErrorTerm")

        assert result is None

    @patch("privfusion.utils.type_classification.SPARQLWrapper")
    def test_find_common_ancestors_success(self, mock_sparql_wrapper):
        """Test finding common ancestors."""
        mock_sparql = MagicMock()
        mock_sparql_wrapper.return_value = mock_sparql

        # Mock responses for both URIs
        mock_results_1 = {
            "results": {
                "bindings": [
                    {"ancestor": {"value": "http://dbpedia.org/ontology/Place"}},
                    {"ancestor": {"value": "http://dbpedia.org/ontology/Location"}},
                ],
            },
        }
        mock_results_2 = {
            "results": {
                "bindings": [
                    {"ancestor": {"value": "http://dbpedia.org/ontology/Place"}},
                    {"ancestor": {"value": "http://dbpedia.org/ontology/PopulatedPlace"}},
                ],
            },
        }

        mock_sparql.queryAndConvert.side_effect = [mock_results_1, mock_results_2]

        result = find_common_ancestors(
            "http://dbpedia.org/resource/City",
            "http://dbpedia.org/resource/Country",
        )

        assert isinstance(result, list)
        assert "http://dbpedia.org/ontology/Place" in result

    @patch("privfusion.utils.type_classification.SPARQLWrapper")
    def test_find_common_ancestors_no_common(self, mock_sparql_wrapper):
        """Test finding common ancestors with no overlap."""
        mock_sparql = MagicMock()
        mock_sparql_wrapper.return_value = mock_sparql

        mock_results_1 = {
            "results": {
                "bindings": [
                    {"ancestor": {"value": "http://dbpedia.org/ontology/Place"}},
                ],
            },
        }
        mock_results_2 = {
            "results": {
                "bindings": [
                    {"ancestor": {"value": "http://dbpedia.org/ontology/Person"}},
                ],
            },
        }

        mock_sparql.queryAndConvert.side_effect = [mock_results_1, mock_results_2]

        result = find_common_ancestors(
            "http://dbpedia.org/resource/City",
            "http://dbpedia.org/resource/Name",
        )

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("privfusion.utils.type_classification.SPARQLWrapper")
    def test_find_common_ancestors_error(self, mock_sparql_wrapper):
        """Test finding common ancestors with SPARQL error."""
        mock_sparql = MagicMock()
        mock_sparql_wrapper.return_value = mock_sparql
        mock_sparql.queryAndConvert.side_effect = Exception("SPARQL error")

        result = find_common_ancestors(
            "http://dbpedia.org/resource/City",
            "http://dbpedia.org/resource/Country",
        )

        assert isinstance(result, list)
        assert len(result) == 0


class TestTypeClassificationEdgeCases:
    """Tests for edge cases in type classification."""

    def test_extract_types_single_column(self):
        """Test extracting types from single column dataframe."""
        df = pd.DataFrame({"col": [1, 2, 3]})

        types = extract_types(df)

        assert len(types) == 1
        assert types[0] == "Numeric"

    def test_extract_types_all_nulls(self):
        """Test extracting types from column with all nulls."""
        df = pd.DataFrame({"col": [None, None, None]})

        types = extract_types(df)

        assert len(types) == 1
        # Should return None for unidentifiable type

    def test_extract_types_mixed_valid_invalid(self):
        """Test extracting types with mix of valid and invalid values."""
        df = pd.DataFrame(
            {
                "mostly_numeric": [1, 2, "three", 4, 5],
            },
        )

        types = extract_types(df)

        assert len(types) == 1
        # Should identify as numeric since majority are numbers
        assert types[0] == "Numeric"

    def test_get_type_mapping_large_dataframe(self):
        """Test type mapping with large dataframe."""
        df = pd.DataFrame({f"col_{i}": range(100) for i in range(10)})

        mappings = get_type_mapping(df)

        assert len(mappings) == 10
        assert all(m == "http://www.w3.org/2001/XMLSchema#Number" for m in mappings)


class TestExtractTypesValueCap:
    """Tests for the per-column unique-value cap (performance)."""

    def test_large_column_still_classified(self):
        """A column with more unique values than the cap is still classified."""
        df = pd.DataFrame({"v": [float(i) for i in range(10000)]})

        assert extract_types(df) == ["Numeric"]
        assert get_type_mapping(df) == ["http://www.w3.org/2001/XMLSchema#Number"]

    def test_large_column_deterministic(self):
        """The sampled subset (and thus the result) is deterministic."""
        df = pd.DataFrame({"v": [float(i) for i in range(10000)]})

        assert extract_types(df) == extract_types(df)

    def test_explicit_max_values(self):
        """max_values can be lowered to force sampling."""
        df = pd.DataFrame({"v": list(range(6000))})

        assert extract_types(df, max_values=1000) == ["Numeric"]

    def test_small_column_unaffected(self):
        """Columns with few unique values are checked in full."""
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})

        assert extract_types(df) == ["Numeric"]
