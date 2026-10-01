import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import requests

# from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
# from langchain_ibm.chat_models import ChatWatsonx
from langchain_ollama import ChatOllama
from litellm import completion
from litellm.exceptions import NotFoundError, RateLimitError, Timeout

logger = logging.getLogger(__name__)


class LLM(ABC):
    @abstractmethod
    def chat(self, messages: list[Any], **kwargs: Any) -> dict[str, str]:
        raise NotImplementedError()


# class WatsonXLLM(LLM):
#     def __init__(self, model_name: str):
#         load_dotenv()

#         parameters = {
#             # GenTextParamsMetaNames.DECODING_METHOD: "sample",
#             GenTextParamsMetaNames.MAX_NEW_TOKENS: 500,
#             GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
#             GenTextParamsMetaNames.TEMPERATURE: 0.5,
#             GenTextParamsMetaNames.TOP_K: 50,
#             GenTextParamsMetaNames.TOP_P: 1,
#         }

#         self._model_name = model_name
#         self._model = ChatWatsonx(
#             model_id=self._model_name,
#             url=SecretStr("https://us-south.ml.cloud.ibm.com"),
#             apikey=SecretStr(os.environ["WATSONX_APIKEY"]),
#             project_id=os.environ["WATSONX_PROJECT_ID"],
#             params=parameters,
#         )

#     def chat(self, messages: list[Any], **kwargs: Any) -> dict[str, str]:
#         response: dict[str, str] = {}
#         try:
#             result = self._model.generate([messages], **kwargs)
#             print(">>>>")
#             pprint(result)
#             print("<<<<")
#             logger.info(result)
#             messages = result.generations[-1]
#             last_message = cast(ChatGeneration, messages[-1]).message
#             response["response"] = str(last_message.content)

#         except Timeout as e:
#             print(f"WatsonX time out error. Check Metrics Dashboard:\n{e.message}")

#         except RateLimitError as e:
#             print(
#                 f"WatsonX user rate limit (8 requests per second) is exceeded\n{e.message}",
#             )

#         except NotFoundError as e:
#             print(f"Selected model not found: {self._model_name}\n{e.message}")

#         except Exception as e:
#             print(e)

#         return response


class LiteLLMLLM(LLM):
    def __init__(self, model_endpoint: str, model_name: str, temperature: float = 0.8):
        self._model_endpoint = model_endpoint
        self._model_name = model_name
        self._temperature = temperature

    def chat(self, messages: list[Any], **kwargs: Any) -> dict[str, str]:
        result: dict[str, str] = {}
        try:
            completion_response = completion(
                model=self._model_name,
                messages=messages,
                api_base=self._model_endpoint,
                temperature=self._temperature,
                **kwargs,
            )
            response = completion_response.choices[0].message.content
            reason = completion_response.choices[0].message.reasoning_content
            result["response"] = response
            result["reason"] = reason

        except Timeout as e:
            print(f"Endpoint time out error. Check Metrics Dashboard:\n{e.message}")

        except RateLimitError as e:
            print(
                f"Endpoint user rate limit is exceeded\n{e.message}",
            )

        except NotFoundError as e:
            print(f"Selected model not found: {self._model_name}\n{e.message}")

        except Exception as e:
            print(e)

        return result


class RitsLLM(LLM):
    def __init__(self, model_endpoint: str, model_name: str):
        self._api_key = os.getenv("RITS_API_KEY", None)
        if self._api_key is None:
            raise OSError("RITS_API_KEY not found in environment.")
        self._model_endpoint = model_endpoint
        self._model_name = model_name

    def chat(self, messages: list[Any], **kwargs: Any) -> dict[str, str]:
        result: dict[str, str] = {}
        try:
            completion_response = completion(
                model="openai/" + self._model_name,
                api_key="fake-key",
                messages=messages,
                api_base=self._model_endpoint + "/v1",
                extra_headers={"RITS_API_KEY": self._api_key},
                **kwargs,
            )
            response = completion_response.choices[0].message.content
            reason = completion_response.choices[0].message.reasoning_content
            result["response"] = response
            result["reason"] = reason

        except Timeout as e:
            print(f"RITS time out error. Check Metrics Dashboard:\n{e.message}")

        except RateLimitError as e:
            print(
                f"RITS user rate limit (1500 requests per minute) is exceeded\n{e.message}",
            )

        except NotFoundError as e:
            print(f"Selected model not found: {self._model_name}\n{e.message}")

        except Exception as e:
            print(e)

        return result


class OllamaLLM(LLM):
    def __init__(
        self,
        model_name: str,
        temperature: int = 0,
        max_tokens: int = 1000,
        reasoning: bool = True,
    ):
        self._model_name = model_name

        self._model = ChatOllama(
            model=self._model_name,
            temperature=temperature,
            num_predict=max_tokens,
        )

    def chat(self, messages: list[Any], **kwargs: Any) -> dict[str, str]:
        reasoning = kwargs.get("reasoning", False)
        result: dict[str, str] = {}
        try:
            response = self._model.invoke(messages, reasoning=reasoning)
            result["response"] = response.content
            result["reason"] = response.additional_kwargs["reasoning_content"]

        except Timeout as e:
            print(f"Ollama time out error:\n{e.message}")

        # except RateLimitError as e:
        #     print(f"RITS user rate limit (1500 requests per minute) is exceeded\n{e.message}")

        except NotFoundError as e:
            print(f"Selected model not found: {self._model_name}\n{e.message}")

        except Exception as e:
            print(">>>> EXCEPTION:")
            print(e)

        return result


class WillmaLLM(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = os.getenv("WILLMA_API_KEY", None)
        if self.api_key is None:
            raise OSError("WILLMA API key not found in environment.")

    def chat(self, messages: list[Any], **kwargs: Any) -> dict[str, str]:
        willma_base_url = "https://api.willma.surf.nl/v0"
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}  # Define python requests headers.

        result: dict[str, str] = {}
        _messages = [{"role": message.type, "content": message.content} for message in messages]

        logger.debug(f"input messages: {_messages}")
        try:
            # TODO: SystemMessage should be serializable but it isn't, so this is a workaround
            _messages = [{"role": message.type, "content": message.content} for message in messages]
            response = requests.post(
                f"{willma_base_url}/chat/completions",
                data=json.dumps({"model": self.model_name, "messages": _messages}),
                headers=headers,
                timeout=42,
            ).json()
            # TODO: configure logging with logging levels (@claude probably)
            # print(response)
            if "error" in response:
                raise ValueError(f'error from willma: {response['message']}, full response: {response}')
            logger.debug(response.json())

            result["response"] = response["choices"][0]["message"]["content"]
            result["reason"] = (
                ""  # TODO: see if we can implement reasoning for willma (need to verify if the model has reasoning)
            )
        except Exception as e:
            print(">>>> EXCEPTION:")
            print(e)

        # print(f'response:{result['response']}')
        return result
