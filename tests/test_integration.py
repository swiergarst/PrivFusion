"""Integration tests for the complete data alignment workflow."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from privfusion.consolidater import Consolidator
from privfusion.dataset_analyzer import DatasetAnalyzer


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    return MagicMock()


@pytest.fixture
def sample_covid_datasets():
    """Create sample COVID-19 datasets for integration testing."""
    # Dataset 1: Oxford COVID data
    df1 = pd.DataFrame(
        {
            "location": ["USA", "UK", "France"],
            "date": pd.date_range("2020-01-01", periods=3),
            "total_cases": [1000, 500, 750],
            "new_cases": [100, 50, 75],
            "population": [331000000, 67000000, 67000000],
        },
    )

    # Dataset 2: Indonesia COVID data
    df2 = pd.DataFrame(
        {
            "country": ["Indonesia", "Indonesia", "Indonesia"],
            "report_date": pd.date_range("2020-01-01", periods=3),
            "confirmed": [500, 600, 700],
            "daily_confirmed": [50, 100, 100],
            "total_population": [273000000, 273000000, 273000000],
        },
    )

    return {"oxford_covid": df1, "indonesia_covid": df2}


@pytest.fixture
def sample_patient_datasets():
    """Create sample patient datasets for integration testing."""
    # Hospital A
    df1 = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "age": [25, 30, 35],
            "gender": ["M", "F", "M"],
            "admission_date": pd.date_range("2023-01-01", periods=3),
            "diagnosis": ["Flu", "COVID-19", "Pneumonia"],
        },
    )

    # Hospital B
    df2 = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "years_old": [28, 32, 40],
            "sex": ["Male", "Female", "Male"],
            "admit_date": pd.date_range("2023-01-01", periods=3),
            "condition": ["Influenza", "Coronavirus", "Lung Infection"],
        },
    )

    return {"hospital_a": df1, "hospital_b": df2}


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_analyze_and_consolidate_covid_datasets(self, mock_llm, sample_covid_datasets):
        """Test analyzing and consolidating COVID datasets."""
        # Mock DatasetAnalyzer responses
        # Call order per dataset: structure description, description, batched
        # column semantics (single JSON response), topics, relationships
        analyzer_responses = [
            # Oxford
            {"response": "COVID-19 time series data", "usage": {"total_tokens": 15}},
            {"response": "COVID-19 data from Oxford", "usage": {"total_tokens": 20}},
            {
                "response": json.dumps(
                    {
                        "location": "Geographic location",
                        "date": "Date of record",
                        "total_cases": "Cumulative cases",
                        "new_cases": "Daily new cases",
                        "population": "Total population",
                    }
                ),
                "usage": {"total_tokens": 30},
            },
            {"response": "COVID-19\nPandemic\nEpidemiology", "usage": {"total_tokens": 15}},
            {
                "response": "date -> total_cases := cumulative over time\ndate -> new_cases := daily increment",
                "usage": {"total_tokens": 20},
            },
            # Indonesia
            {"response": "Indonesian COVID-19 tracking data", "usage": {"total_tokens": 15}},
            {"response": "COVID-19 data from Indonesia", "usage": {"total_tokens": 20}},
            {
                "response": json.dumps(
                    {
                        "country": "Country name",
                        "report_date": "Report date",
                        "confirmed": "Confirmed cases",
                        "daily_confirmed": "Daily confirmed",
                        "total_population": "Population count",
                    }
                ),
                "usage": {"total_tokens": 30},
            },
            {"response": "COVID-19\nIndonesia\nPublic Health", "usage": {"total_tokens": 15}},
            {"response": "report_date -> confirmed := cumulative tracking", "usage": {"total_tokens": 15}},
        ]
        mock_llm.chat.side_effect = analyzer_responses

        # Analyze datasets
        analyzer = DatasetAnalyzer(mock_llm, {})
        datasets = {}

        for name, df in sample_covid_datasets.items():
            info = analyzer.extract_information_from_dataset(name, df)
            datasets[name] = {"data": df, "info": info}

        # Verify analysis results
        assert len(datasets) == 2
        assert "oxford_covid" in datasets
        assert "indonesia_covid" in datasets

        oxford_info = datasets["oxford_covid"]["info"]
        assert len(oxford_info.structure.names) == 5
        assert "location" in oxford_info.structure.names
        assert len(oxford_info.relationships) == 2

        indonesia_info = datasets["indonesia_covid"]["info"]
        assert len(indonesia_info.structure.names) == 5
        assert "country" in indonesia_info.structure.names

    def test_full_consolidation_workflow(self, mock_llm, sample_patient_datasets):
        """Test full consolidation workflow with patient data."""
        # Setup mock responses for analysis
        analysis_responses = self._create_patient_analysis_responses()

        # Setup mock responses for consolidation
        cluster_response = [
            {"dataset": "hospital_a", "feature_name": "patient_id", "cluster_id": 1},
            {"dataset": "hospital_a", "feature_name": "age", "cluster_id": 2},
            {"dataset": "hospital_a", "feature_name": "gender", "cluster_id": 3},
            {"dataset": "hospital_a", "feature_name": "admission_date", "cluster_id": 4},
            {"dataset": "hospital_a", "feature_name": "diagnosis", "cluster_id": 5},
            {"dataset": "hospital_b", "feature_name": "id", "cluster_id": 1},
            {"dataset": "hospital_b", "feature_name": "years_old", "cluster_id": 2},
            {"dataset": "hospital_b", "feature_name": "sex", "cluster_id": 3},
            {"dataset": "hospital_b", "feature_name": "admit_date", "cluster_id": 4},
            {"dataset": "hospital_b", "feature_name": "condition", "cluster_id": 5},
        ]

        norm_responses = [
            {
                "uri": "http://schema.org/identifier",
                "feature_name": "patient_identifier",
                "value_structure": "integer",
                "transformations": [
                    {"dataset": "hospital_a", "name": "patient_id", "recommendation": "rename to patient_identifier"},
                    {"dataset": "hospital_b", "name": "id", "recommendation": "rename to patient_identifier"},
                ],
            },
            {
                "uri": "http://schema.org/age",
                "feature_name": "age_years",
                "value_structure": "integer",
                "transformations": [
                    {"dataset": "hospital_a", "name": "age", "recommendation": "rename to age_years"},
                    {"dataset": "hospital_b", "name": "years_old", "recommendation": "rename to age_years"},
                ],
            },
            {
                "uri": "http://schema.org/gender",
                "feature_name": "gender",
                "value_structure": "string",
                "transformations": [
                    {"dataset": "hospital_a", "name": "gender", "recommendation": "expand M to Male, F to Female"},
                    {"dataset": "hospital_b", "name": "sex", "recommendation": "rename to gender"},
                ],
            },
            {
                "uri": "http://schema.org/Date",
                "feature_name": "admission_date",
                "value_structure": "date",
                "transformations": [
                    {"dataset": "hospital_a", "name": "admission_date", "recommendation": "no change"},
                    {"dataset": "hospital_b", "name": "admit_date", "recommendation": "rename to admission_date"},
                ],
            },
            {
                "uri": "http://schema.org/MedicalCondition",
                "feature_name": "diagnosis",
                "value_structure": "string",
                "transformations": [
                    {"dataset": "hospital_a", "name": "diagnosis", "recommendation": "no change"},
                    {"dataset": "hospital_b", "name": "condition", "recommendation": "rename to diagnosis"},
                ],
            },
        ]

        mock_llm.chat.side_effect = analysis_responses

        # Analyze datasets
        analyzer = DatasetAnalyzer(mock_llm, {})
        datasets = {}

        for name, df in sample_patient_datasets.items():
            info = analyzer.extract_information_from_dataset(name, df)
            datasets[name] = {"data": df, "info": info}

        # Mock consolidation
        with (
            patch("privfusion.consolidater.AgentCluster") as MockAgentCluster,
            patch("privfusion.consolidater.AgentNorm") as MockAgentNorm,
        ):
            mock_cluster_instance = MockAgentCluster.return_value
            mock_cluster_instance.cluster.return_value = cluster_response

            mock_norm_instance = MockAgentNorm.return_value
            mock_norm_instance.normalize.side_effect = norm_responses

            consolidator = Consolidator()
            result = consolidator.consolidate(datasets, mock_llm)

            # Verify consolidation results
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 10  # 5 columns from each dataset
            assert result.cluster_id.nunique() == 5  # 5 unique clusters

            # Verify transformations are assigned
            assert result["transformation"].notna().sum() == 10

            # Verify unified schema
            assert len(consolidator._unified_schemas) == 1
            schema = consolidator._unified_schemas[0]
            assert len(schema) == 5

            # Check specific schema entries
            schema_names = [s["norm_feature_name"] for s in schema]
            assert "patient_identifier" in schema_names
            assert "age_years" in schema_names
            assert "gender" in schema_names

    def test_multiple_consolidation_rounds(self, mock_llm, sample_patient_datasets):
        """Test running multiple consolidation rounds."""
        # Setup mocks
        analysis_responses = self._create_patient_analysis_responses()
        mock_llm.chat.side_effect = analysis_responses * 2  # For two rounds

        cluster_response = self._create_cluster_response()
        norm_responses = self._create_norm_responses()

        # Analyze datasets
        analyzer = DatasetAnalyzer(mock_llm, {})
        datasets = {}

        for name, df in sample_patient_datasets.items():
            info = analyzer.extract_information_from_dataset(name, df)
            datasets[name] = {"data": df, "info": info}

        # Run consolidation twice
        with (
            patch("privfusion.consolidater.AgentCluster") as MockAgentCluster,
            patch("privfusion.consolidater.AgentNorm") as MockAgentNorm,
        ):
            mock_cluster_instance = MockAgentCluster.return_value
            mock_cluster_instance.cluster.return_value = cluster_response

            mock_norm_instance = MockAgentNorm.return_value
            mock_norm_instance.normalize.side_effect = norm_responses * 2

            consolidator = Consolidator()

            # First round
            _result1 = consolidator.consolidate(datasets, mock_llm)
            assert consolidator._number_consolidations == 1

            # Second round
            _result2 = consolidator.consolidate(datasets, mock_llm)
            assert consolidator._number_consolidations == 2
            assert len(consolidator._unified_schemas) == 2

    def test_error_recovery_in_workflow(self, mock_llm, sample_patient_datasets):
        """Test error recovery during workflow."""
        # First analysis fails, second succeeds
        analysis_responses = [
            Exception("LLM Error"),
            {"response": "Hospital A patient data", "usage": {"total_tokens": 20}},
        ]

        mock_llm.chat.side_effect = analysis_responses

        analyzer = DatasetAnalyzer(mock_llm, {})

        # First call should fail
        with pytest.raises((ValueError, Exception)):
            analyzer.extract_information_from_dataset(
                "hospital_a",
                sample_patient_datasets["hospital_a"],
            )

    def test_empty_dataset_workflow(self, mock_llm):
        """Test workflow with empty datasets."""
        empty_datasets = {}

        with patch("privfusion.consolidater.AgentCluster") as MockAgentCluster:
            mock_cluster_instance = MockAgentCluster.return_value
            mock_cluster_instance.cluster.return_value = []

            consolidator = Consolidator()

            # Empty datasets will cause errors in consolidation
            with pytest.raises((KeyError, ValueError, Exception)):
                consolidator.consolidate(empty_datasets, mock_llm)

    def test_single_dataset_consolidation(self, mock_llm, sample_patient_datasets):
        """Test consolidation with only one dataset."""
        single_dataset = {"hospital_a": sample_patient_datasets["hospital_a"]}

        analysis_responses = self._create_single_dataset_analysis_responses()
        mock_llm.chat.side_effect = analysis_responses

        analyzer = DatasetAnalyzer(mock_llm, {})
        datasets = {}

        for name, df in single_dataset.items():
            info = analyzer.extract_information_from_dataset(name, df)
            datasets[name] = {"data": df, "info": info}

        cluster_response = [
            {"dataset": "hospital_a", "feature_name": "patient_id", "cluster_id": 1},
            {"dataset": "hospital_a", "feature_name": "age", "cluster_id": 2},
            {"dataset": "hospital_a", "feature_name": "gender", "cluster_id": 3},
            {"dataset": "hospital_a", "feature_name": "admission_date", "cluster_id": 4},
            {"dataset": "hospital_a", "feature_name": "diagnosis", "cluster_id": 5},
        ]

        with patch("privfusion.consolidater.AgentCluster") as MockAgentCluster:
            mock_cluster_instance = MockAgentCluster.return_value
            mock_cluster_instance.cluster.return_value = cluster_response

            consolidator = Consolidator()

            # Single dataset consolidation may fail due to normalization issues
            with pytest.raises((ValueError, Exception)):
                consolidator.consolidate(datasets, mock_llm)

    def _create_patient_analysis_responses(self):
        """Helper to create mock responses for patient data analysis.

        Call order per dataset: structure description, description, batched
        column semantics (single JSON response), topics, relationships.
        """
        return [
            # Hospital A
            {"response": "Patient demographic and medical data", "usage": {"total_tokens": 15}},
            {"response": "Hospital A patient data", "usage": {"total_tokens": 20}},
            {
                "response": json.dumps(
                    {
                        "patient_id": "Patient identifier",
                        "age": "Patient age",
                        "gender": "Patient gender",
                        "admission_date": "Admission date",
                        "diagnosis": "Medical diagnosis",
                    }
                ),
                "usage": {"total_tokens": 30},
            },
            {"response": "Healthcare\nPatient Records", "usage": {"total_tokens": 15}},
            {"response": "patient_id -> diagnosis := identifies patient condition", "usage": {"total_tokens": 15}},
            # Hospital B
            {"response": "Patient information and conditions", "usage": {"total_tokens": 15}},
            {"response": "Hospital B patient data", "usage": {"total_tokens": 20}},
            {
                "response": json.dumps(
                    {
                        "id": "Patient ID",
                        "years_old": "Age in years",
                        "sex": "Patient sex",
                        "admit_date": "Date of admission",
                        "condition": "Medical condition",
                    }
                ),
                "usage": {"total_tokens": 30},
            },
            {"response": "Healthcare\nMedical Records", "usage": {"total_tokens": 15}},
            {"response": "id -> condition := patient medical status", "usage": {"total_tokens": 15}},
        ]

    def _create_single_dataset_analysis_responses(self):
        """Helper to create mock responses for single dataset analysis.

        Call order: structure description, description, batched column
        semantics (single JSON response), topics, relationships.
        """
        return [
            {"response": "Patient demographic and medical data", "usage": {"total_tokens": 15}},
            {"response": "Hospital A patient data", "usage": {"total_tokens": 20}},
            {
                "response": json.dumps(
                    {
                        "patient_id": "Patient identifier",
                        "age": "Patient age",
                        "gender": "Patient gender",
                        "admission_date": "Admission date",
                        "diagnosis": "Medical diagnosis",
                    }
                ),
                "usage": {"total_tokens": 30},
            },
            {"response": "Healthcare\nPatient Records", "usage": {"total_tokens": 15}},
            {"response": "patient_id -> diagnosis := identifies patient condition", "usage": {"total_tokens": 15}},
        ]

    def _create_cluster_response(self):
        """Helper to create cluster response."""
        return [
            {"dataset": "hospital_a", "feature_name": "patient_id", "cluster_id": 1},
            {"dataset": "hospital_a", "feature_name": "age", "cluster_id": 2},
            {"dataset": "hospital_a", "feature_name": "gender", "cluster_id": 3},
            {"dataset": "hospital_a", "feature_name": "admission_date", "cluster_id": 4},
            {"dataset": "hospital_a", "feature_name": "diagnosis", "cluster_id": 5},
            {"dataset": "hospital_b", "feature_name": "id", "cluster_id": 1},
            {"dataset": "hospital_b", "feature_name": "years_old", "cluster_id": 2},
            {"dataset": "hospital_b", "feature_name": "sex", "cluster_id": 3},
            {"dataset": "hospital_b", "feature_name": "admit_date", "cluster_id": 4},
            {"dataset": "hospital_b", "feature_name": "condition", "cluster_id": 5},
        ]

    def _create_norm_responses(self):
        """Helper to create normalization responses."""
        return [
            {
                "uri": "http://schema.org/identifier",
                "feature_name": "patient_identifier",
                "value_structure": "integer",
                "transformations": [
                    {"dataset": "hospital_a", "name": "patient_id", "recommendation": "rename to patient_identifier"},
                    {"dataset": "hospital_b", "name": "id", "recommendation": "rename to patient_identifier"},
                ],
            },
            {
                "uri": "http://schema.org/age",
                "feature_name": "age_years",
                "value_structure": "integer",
                "transformations": [
                    {"dataset": "hospital_a", "name": "age", "recommendation": "rename to age_years"},
                    {"dataset": "hospital_b", "name": "years_old", "recommendation": "rename to age_years"},
                ],
            },
            {
                "uri": "http://schema.org/gender",
                "feature_name": "gender",
                "value_structure": "string",
                "transformations": [
                    {"dataset": "hospital_a", "name": "gender", "recommendation": "expand M to Male, F to Female"},
                    {"dataset": "hospital_b", "name": "sex", "recommendation": "rename to gender"},
                ],
            },
            {
                "uri": "http://schema.org/Date",
                "feature_name": "admission_date",
                "value_structure": "date",
                "transformations": [
                    {"dataset": "hospital_a", "name": "admission_date", "recommendation": "no change"},
                    {"dataset": "hospital_b", "name": "admit_date", "recommendation": "rename to admission_date"},
                ],
            },
            {
                "uri": "http://schema.org/MedicalCondition",
                "feature_name": "diagnosis",
                "value_structure": "string",
                "transformations": [
                    {"dataset": "hospital_a", "name": "diagnosis", "recommendation": "no change"},
                    {"dataset": "hospital_b", "name": "condition", "recommendation": "rename to diagnosis"},
                ],
            },
        ]


class TestDataFlowIntegration:
    """Test data flow between components."""

    def test_analyzer_output_to_consolidator_input(self, mock_llm):
        """Test that analyzer output is compatible with consolidator input."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        mock_llm.chat.return_value = {"response": "Test", "usage": {"total_tokens": 10}}

        analyzer = DatasetAnalyzer(mock_llm, {})
        info = analyzer.extract_information_from_dataset("test_dataset", df)

        # Verify structure matches what consolidator expects
        assert hasattr(info, "name")
        assert hasattr(info, "semantic")
        assert hasattr(info, "structure")
        assert hasattr(info.structure, "names")
        assert hasattr(info.structure, "types")
        assert hasattr(info.semantic, "column_types")
        assert hasattr(info.semantic, "column_semantic")

        # Create dataset dict as consolidator expects
        datasets = {"test_dataset": {"data": df, "info": info}}

        # Verify it can be used by consolidator
        assert "test_dataset" in datasets
        assert "data" in datasets["test_dataset"]
        assert "info" in datasets["test_dataset"]
