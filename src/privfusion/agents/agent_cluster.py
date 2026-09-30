import json
import logging
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
from tenacity import retry, stop_after_attempt, wait_fixed

from privfusion.agents.llms import LLM
from privfusion.utils.utils import print_colored_dict

logger = logging.getLogger(__name__)


class AgentCluster:
    def __init__(self, llm: LLM, system_prompt: str, **kwargs: Any):
        self._llm = llm
        self._system_prompt = system_prompt
        self._kwargs = kwargs

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    def cluster(self, datasets: dict[str, Any]) -> dict[str, Any]:
        dataset_context = pd.DataFrame(columns=["dataset"])
        for dataset_name in datasets:
            ds = datasets[dataset_name]["info"]
            temp = pd.DataFrame(columns=["dataset"])
            temp["feature_name"] = ds.structure.names
            temp["feature_type"] = ds.structure.types
            temp["feature_semantic_type"] = ds.semantic.column_types
            temp["feature_semantic_desc"] = ds.semantic.column_semantic
            temp["dataset"] = ds.name
            dataset_context = pd.concat([dataset_context, temp], axis=0)

        messages = [SystemMessage(self._system_prompt), HumanMessage(dataset_context.to_json(orient="records"))]
        result = self._llm.chat(messages=messages, **self._kwargs)
        print_colored_dict(result)
        try:
            clusters = json.loads(result["response"])
        except Exception as e:
            logger.error("Exception converting cluster output to json.")
            raise ValueError("Exception converting cluster output to json.") from e

        cf = pd.DataFrame(clusters)

        # validate that all the features mentioned, do exist in relevant dataset
        for dataset_name in datasets:
            ds = datasets[dataset_name]["info"]
            # check length
            cluster_feature_names = cf[cf.dataset == ds.name].feature_name.sort_values().tolist()
            ds_feauture_names = sorted(ds.structure.names)
            # seems as thought the cluster_feature_names hallucinates a few features -> potentially just drop any
            # hallucinated features

            for _, row in cf[cf.dataset == ds.name].reset_index(drop=True).iterrows():
                if row.feature_name not in ds_feauture_names:
                    cf = cf[~((cf.dataset == ds.name) & (cf.feature_name == row.feature_name))].reset_index(drop=True)
            if len(cluster_feature_names) != len(ds_feauture_names):
                logger.error(
                    "Cluster agent didn't generate correct number of features for dataset '{ds}'.",
                )
                raise ValueError(
                    f"Cluster agent didn't generate correct number of features for dataset '{ds}'.",
                )
            if cluster_feature_names != ds_feauture_names:
                logger.error(
                    f"Cluster agent didn't generate correctly named features for dataset '{ds}'.",
                )
                raise ValueError(
                    f"Cluster agent didn't generate correctly named features for dataset '{ds}'.",
                )

        return clusters
