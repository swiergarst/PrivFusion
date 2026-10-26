import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any

import requests

# from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
# from langchain_ibm.chat_models import ChatWatsonx
from langchain_ollama import ChatOllama
from litellm import completion
from litellm.exceptions import NotFoundError, RateLimitError, Timeout

# setup a separate logger that logs all input and output to and from the LLM
llmLogger = logging.getLogger("fedHarmonizationLogger")

# llmLogger = logging.getLogger("llmLogger")
# llmLogger.setLevel(logging.INFO)
# llmLogger.propagate = False

# handler = logging.FileHandler(f"logs/llm_logs_{time.time()}.log")
# llmLogger.addHandler(handler)


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
            llmLogger.error(f"Endpoint time out error. Check Metrics Dashboard:\n{e.message}")

        except RateLimitError as e:
            llmLogger.error(
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
    def __init__(self, model_name: str, max_retries: int = 3, timeout: float = 120.0):
        self.model_name = model_name
        self.api_key = os.getenv("WILLMA_API_KEY", None)
        if self.api_key is None:
            raise OSError("WILLMA API key not found in environment.")
        self._max_retries = max_retries
        self._timeout = timeout

    def chat(self, messages: list[Any], **kwargs: Any) -> dict[str, str]:
        willma_base_url = "https://api.willma.surf.nl/v0"
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}  # Define python requests headers.

        # _messages = [{"role": message.type, "content": message.content} for message in messages]

        # TODO: SystemMessage should be serializable but it isn't, so this is a workaround
        _messages = []
        for message in messages:
            new_message = {"role": message.type, "content": message.content}
            # workaround for mismatch between langchain and willma
            if new_message["role"] == "human":
                new_message["role"] = "user"
            _messages.append(new_message)

        payload: dict[str, Any] = {"model": self.model_name, "messages": _messages}
        if kwargs.get("response_format"):
            payload["response_format"] = kwargs["response_format"]

        llmLogger.info(f"input messages: {_messages}")

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            started = time.time()
            try:
                response_obj = requests.post(
                    f"{willma_base_url}/chat/completions",
                    data=json.dumps(payload),
                    headers=headers,
                    timeout=self._timeout,
                )
                if response_obj.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {response_obj.status_code} from willma")
                response = response_obj.json()
            except (requests.RequestException, ValueError) as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = 2.0**attempt + random.uniform(0, 1)
                    llmLogger.warning(
                        f"willma request failed (attempt {attempt}/{self._max_retries}): {e}; retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                llmLogger.error(f"willma request failed after {self._max_retries} attempts: {e}")
                raise RuntimeError(f"willma request failed after {self._max_retries} attempts: {e}") from e

            if "error" in response:
                last_error = ValueError(f"error from willma, full response: {response}")
                if attempt < self._max_retries:
                    delay = 2.0**attempt + random.uniform(0, 1)
                    llmLogger.warning(
                        f"willma returned an error (attempt {attempt}/{self._max_retries}); retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                llmLogger.error(f"willma returned an error after {self._max_retries} attempts: {response}")
                raise last_error

            llmLogger.info(f"response: {response}")
            try:
                content = response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                last_error = ValueError(f"no content in willma response: {response}")
                if attempt < self._max_retries:
                    delay = 2.0**attempt + random.uniform(0, 1)
                    llmLogger.warning(
                        f"willma response had no content (attempt {attempt}/{self._max_retries}); retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                llmLogger.error(f"willma response had no content after {self._max_retries} attempts: {response}")
                raise last_error from e

            llmLogger.info(f"willma call took {time.time() - started:.1f}s")
            return {
                "response": content,
                "reason": (
                    ""  # TODO: see if we can implement reasoning for willma (need to verify if the model has reasoning)
                ),
            }

        # Unreachable: every attempt either returns, raises, or continues
        raise last_error if last_error is not None else RuntimeError("willma chat retries exhausted")
