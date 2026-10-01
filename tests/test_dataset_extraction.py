from unittest.mock import MagicMock

import pandas as pd
from faker import Faker

from privfusion.data_models import DatasetInformation, SemanticInformation, StructuralInformation
from privfusion.dataset_analyzer import DatasetAnalyzer


def test_structure() -> None:
    dataset = pd.DataFrame(
        {
            "float": [1.0],
            "int": [1],
            "datetime": [pd.Timestamp("20180310")],
            "string": ["foo"],
        },
    )

    # Create mock LLM and config
    mock_llm = MagicMock()
    mock_config = {}

    analyzer = DatasetAnalyzer(mock_llm, mock_config)
    structure = analyzer.extract_information_from_dataset("my_dataset", dataset).structure

    assert structure is not None
    assert len(structure.names) == len(structure.types)


def test_semantic(faker: Faker) -> None:
    dataset = pd.DataFrame(
        {
            "zipcode": [faker.zipcode() for _ in range(100)],
            "patient name": [faker.name() for _ in range(100)],
        },
    )

    # Create mock LLM and config
    mock_llm = MagicMock()
    mock_config = {}

    analyzer = DatasetAnalyzer(mock_llm, mock_config)
    semantic = analyzer.extract_information_from_dataset("my_dataset", dataset).semantic

    assert semantic is not None


def test_store_dataset_information() -> None:
    mock_semantics = SemanticInformation(description="", column_types=[], column_semantic=[], dataset_topics=[])
    mock_structural = StructuralInformation(names=[], types=[], constraints=[], description="")
    mock_info = DatasetInformation(name="mock", semantic=mock_semantics, structure=mock_structural, relationships=[])

    mock_info.write("test_dataset.json")

    # TODO: have a different initialization from the original
    mock_info_recovered = DatasetInformation(
        name="mock", semantic=mock_semantics, structure=mock_structural, relationships=[]
    ).read("test_dataset.json")

    assert mock_info.__eq__(mock_info_recovered)
