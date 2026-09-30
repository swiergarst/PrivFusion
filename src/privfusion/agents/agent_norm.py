import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
from tenacity import retry, stop_after_attempt, wait_fixed

from privfusion.agents.llms import LLM
from privfusion.utils.utils import print_colored_dict

logger = logging.getLogger(__name__)


class AgentNorm:
    def __init__(self, llm: LLM, system_prompt: str, **kwargs: Any):
        self._llm = llm
        self._system_prompt = system_prompt
        self._kwargs = kwargs

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def normalize(self, cluster_context: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [
            SystemMessage(self._system_prompt),
            HumanMessage(str(cluster_context)),
            # {"role": "system", "content": self._system_prompt},
            # {"role": "user", "content": str(cluster_context)},
        ]

        result = self._llm.chat(messages=messages, **self._kwargs)
        print_colored_dict(result)
        try:
            normalized_cluster = json.loads(result["response"])
        except Exception as e:
            logger.error("Exception converting normalized output to json.")

            print(result["response"])
            raise ValueError("Exception converting normalized output to json.") from e

        return normalized_cluster
