class DatasetAnalyzerPrompts:
    DESCRIPTION_SYSTEM_PROMPT = """I have a tabular dataset contisting of various columns. Given the column names and some examples as per (SAMPLE) describe the overall semantic of the dataset in a formal style. Only return the discorsive description."""
    COLUMN_SEMANTIC_SYSTEM_PROMPT = """Given the description of a dataset (DESCRIPTION), the column type (COLUMN_TYPE) describe the semantic of the column (COLUMN) in a formal style. Only return the discorsive description."""
    TOPICS_SYSTEM_PROMPT = """Given the dataset descripion (DESCRIPTION) and the sample (SAMPLE) list topics of the dataset. Respond with a list of topics one per line only"""
    RELATIONSHIPS_SYSTEM_PROMPT = """Given the dataset in (SAMPLE) highlight relationships between columns in the following format: {column_name} -> {column_name} := {explanation}. Do not provide any comment."""
    STRUCTURE_DESCRIPTION_SYSTEM_PROMPT = """Given a tabular datasets consisting of columns (COLUMN_NAMES) and some examples as per (SAMPLE), please describe the structure of the dataset to the best of your understanding. Only return the discorsive description."""


class ConsolidatorPrompts:
    CLUSTER_AGENT_SYSTEM_PROMPT = """I have a number of datasets, each with a list of features. Each feature has a feature_name, feature_type and feature_semantic_type.
   Many of the datasets contain overlapping features. I wish to converge to a single dataset. In order to further analyse which features should be combined
   I want you to group features that are semantically similar and should be candidates for merging into the same singular feature. A cluster should not contain
   more than 1 feature from the same dataset. If no cluster can be found for a feature (e.g. due to a feature identified as not common to both datasets),
   add the feature to cluster_id 9999.
   The datasets will be given to you in json form as follows:
    [
      {
        "dataset": "name_of_dataset",
        "feature_name": "feature_name",
        "feature_type": "feature_type",
        "feature_semantic_type": "feature_semantic_type"
      },
      {
        "dataset": "name_of_dataset",
        "feature_name": "feature_name",
        "feature_type": "feature_type",
        "feature_semantic_type": "feature_semantic_type"
      },
      ...
    ]

    Your output should be in JSON format (no comments!) as follows:

    [
      {
        "cluster_id": 1,
        "feature_name": "feature_name",
        "dataset": "name_of_dataset"
      }
    ]"""

    NORM_AGENT_SYSTEM_PROMPT = """I have a number of dbpedia URIs that describe similar features across a variety of datasets.
    I want to converge on a single dbpedia URI that describes the majority of datasets whilst maintaining granularity.
    Some dataset features will be more granular than others. I wish to implement majority rule for the level of granularity.
    If the majority of features are low level entities (e.g. subclass of a higher feature), a low level entity should be selected
    as the URI (or the next ancestor of low level entity).

    Here are the rules:

    - The goal is to create a unified representation that satisfies the majority of datasets.
    - There should be no overlapping data values in the feature when unified.
    - Do not permit multiple granularity of data to be in the unified representation, choose one.
    - Do not permit user choices in the value_structure returned. Be decisive and concise.
    - In a tie-break, favour the higher level.
    - If sample_data does not reflect the entity correctly, reason using the correct entity and recommend transformation.
    - I want to normalize to a majority-rules level entity (which could be one of the listed URIs if it includes ancestor of descendants of that URI).
    - If most features are a lower level of granularity, then a lower level entity should be selected as the URI (or a close ancestor to lower level entity).
    - The selected dbpedia URI may be a URI of one of the datasets (if majority of others are ancestors or descendants of that URI).
    - Majority rules - if the majority of URIs are subclasses, then a subclass entity should be selected.
    - Assume there is only high and low levels when assessing the majority.
    - Favour reusing a URI provided by the user, once it fits the other rules - consider all URIs in the list.
    - For any of the feature URIs presented, if they do not fit neatly into the selected URI, include a recommended (e.g. data transformation) to apply to the dataset owner to align.
    - Use the selected uri to create the feature name
    - The value_structure contains the values permitted for the feature - it must strictly abide by the URI selected.
    - Prefer not to use full URIs as values of a feature, but can use the entity name

    The user query will be in the form:
    [
        {
            "name": "name_1",
            "dataset": "dataset_1",
            "uri": "uri_1",
            "data": [sample_data_1, sample_data_2, sample_data_3]
        },
        {
            "name": "name_2",
            "dataset": "dataset_2",
            "uri": "uri_2",
            "data": [sample_data_1, sample_data_2, sample_data_3]
        },
    ]

    Your output should be in json format as follows:
    {"uri": "uri",
    "feature_name": "snake_case_feature_name",
    "value_structure": "explanation of the feature and what are valid values",
    "transformations": [
        {
            "name": "name_1",
            "dataset": "dataset_1",
            "recommendation": "recommended transformation to apply such that feature aligns to uri, feature_name and value_structure."
        },
        {
            "name": "name_2",
            "dataset": "dataset_2",
            "recommendation": "recommended transformation to apply such that feature aligns to uri, feature_name and value_structure."
        },
    ]"""
