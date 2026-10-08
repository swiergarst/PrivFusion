import logging
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed

from privfusion.agents.agent_cluster import AgentCluster
from privfusion.agents.agent_norm import AgentNorm
from privfusion.agents.default_system_prompts import ConsolidatorPrompts
from privfusion.agents.llms import LLM

logger = logging.getLogger(__name__)


class Consolidator:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._number_consolidations = 0
        self._unified_schemas = []
        if config is not None:
            self.config = config
        else:
            self.config = {}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def consolidate(self, datasets: dict[str, Any], llm: LLM, **kwargs: Any):
        """
        Executes consolidation on the datasets
        - first runs agent_cluster to cluster the likely column attributes
        - secondly runs agent_norm to normalize the identified columns
        - [optional] save transformations to database of all transformations
        - saves the working data schema
        - [optional] generate synthetic data using the schema
        - map the new schema back to the original datasets
        - finally returns feature transformations recommendations per party
        """

        logger.info("Clustering dataset features.")
        agent_cluster = AgentCluster(
            llm,
            self.config.get("cluster", {}).get("system_prompt", ConsolidatorPrompts.CLUSTER_AGENT_SYSTEM_PROMPT),
            kwargs,
        )
        result = agent_cluster.cluster(datasets)

        cf = pd.DataFrame(result)
        cf = cf.sort_values("cluster_id").reset_index(drop=True)
        logger.info(f"Identified {cf.cluster_id.nunique()} unique clusters.")

        logger.info("Normalizing clusters.")
        for cluster_id in cf[cf.cluster_id != 9999].reset_index(drop=True).cluster_id.unique():
            logger.info(f"-- normalizing cluster {cluster_id}")
            cluster_context = []
            for _, row in cf[cf.cluster_id == cluster_id].reset_index(drop=True).iterrows():
                feature_name = row.feature_name
                dataset_name = row.dataset
                dataset_info = datasets[dataset_name]["info"]
                feature_idx = dataset_info.structure.names.index(feature_name)
                uri = dataset_info.semantic.column_types[feature_idx]
                example_data = datasets[dataset_name]["data"][feature_name].unique()[:5]
                cluster_context.append(
                    {
                        "name": feature_name,
                        "dataset": dataset_name,
                        "uri": uri,
                        "example_data": example_data,
                    },
                )

            agent_norm = AgentNorm(
                llm,
                self.config.get("normalize", {}).get("system_prompt", ConsolidatorPrompts.NORM_AGENT_SYSTEM_PROMPT),
                kwargs,
            )
            result = agent_norm.normalize(cluster_context)

            # map the transformation recommendations back to dataset -> column
            cf.loc[cf.cluster_id == cluster_id, "norm_uri"] = result["uri"]
            cf.loc[cf.cluster_id == cluster_id, "norm_feature_name"] = result["feature_name"]
            cf.loc[cf.cluster_id == cluster_id, "norm_value_structure"] = result["value_structure"]
            for item in result["transformations"]:
                cf.loc[
                    (cf.cluster_id == cluster_id) & (cf.dataset == item["dataset"]) & (cf.feature_name == item["name"]),
                    "transformation",
                ] = item["recommendation"]

        logger.info("Updating the unified schema")
        schema = (
            cf[["norm_feature_name", "norm_value_structure", "norm_uri"]]
            .dropna()
            .drop_duplicates()
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        self._unified_schemas.append(schema)
        self._number_consolidations += 1

        return cf
