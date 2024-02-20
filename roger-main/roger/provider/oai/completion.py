from typing import Union, Any, Callable, Optional, List, Dict, Tuple, Iterator

from openai import OpenAI, AsyncOpenAI

from roger.provider.oai.error import oai_error
from roger.provider.oai.keys import get_key_params
from roger.provider.status import ResponseStatus


class OaiCompletion:
    __model_name: str = "gpt-3.5-turbo-0613"
    __create: Callable
    __acreate: Callable

    def __init__(
        self,
        api_key: str,
        org_key: str,
        model_name: Optional[str] = None,
        max_retries: int = 2,
        timeout_ms: int = 20,
    ):
        # Get OPENAI_API_KEY and OPENAI_ORG_KEY
        key_params = get_key_params(api_key=api_key, org_key=org_key)

        if model_name:
            self.__model_name = model_name

        self.__client_options = {
            "max_retries": max_retries,
            "timeout": timeout_ms * 1000,
        }

        self.__create = OpenAI(**key_params).with_options(**self.__client_options).chat.completions.create
        self.__acreate = AsyncOpenAI(**key_params).with_options(**self.__client_options).chat.completions.create

    def __set_params(self, **kwargs: Any) -> Dict[str, Union[str, int, float, List[str]]]:
        params = {"model": self.__model_name}

        if "messages" in kwargs:
            params["messages"] = kwargs.get("messages")
        else:
            raise ValueError("Message parameter must be inserted")

        return {
            **params,
            "temperature": kwargs.get("temperature", 1.0),
            "max_tokens": kwargs.get("max_tokens", 256),
            "stop": kwargs.get("stop_words", []),
            "stream": kwargs.get("stream", False),
        }

    @oai_error
    def call(self, **params: Any) -> Tuple[ResponseStatus, Optional[str]]:
        """
        입력에 맞게 응답함(동기식)

        :param messages: 메시지
        :param temperature: randomness
        :param max_tokens: 최대 생성 길이
        :param stop: stop word
        :return: 생성된 문자열
        """

        return ResponseStatus.SUCCESS, self.__create(**self.__set_params(**params)).choices[0].message.content

    @oai_error
    async def acall(self, **params: Any) -> tuple[ResponseStatus, Optional[str]]:
        """
        입력에 맞게 응답함(동기식)

        :param messages: 메시지
        :param temperature: randomness
        :param max_tokens: 최대 생성 길이
        :param stop: stop word
        :return: 생성된 문자열
        """

        return ResponseStatus.SUCCESS, (await self.__acreate(**self.__set_params(**params))).choices[0].message.content

    @oai_error
    def stream(self, **params: Any) -> Iterator[Tuple[ResponseStatus, str]]:
        """
        입력에 맞게 응답함(동기식)

        :param messages: 메시지
        :param temperature: randomness
        :param max_tokens: 최대 생성 길이
        :param stop: stop word
        :return: 생성된 문자열
        """

        params["stream"] = True

        for part in self.__create(**self.__set_params(**params)):
            yield ResponseStatus.SUCCESS, (part.choices[0].delta.content or "")

    #
    # async def astream(self, **params: Any) -> AsyncIterator[Tuple[ResponseStatus, str]]:
    #     """
    #     입력에 맞게 스트리밍 방식으로 응답함(비동기식)
    #
    #     :param messages: 메시지
    #     :param temperature: randomness
    #     :param max_tokens: 최대 생성 길이
    #     :param stop: stop word
    #     :return: 생성된 문자열
    #     """
    #
    #     params["stream"] = True
    #
    #     try:
    #         stream = await self.__acreate(**self.__set_params(**params))
    #
    #         for part in stream:
    #             yield ResponseStatus.SUCCESS, (part.choices[0].delta.content or "")
    #
    #     except MANAGEABLE_ERROR:
    #         yield ResponseStatus.ROUTE, ""
    #     except CRITICAL_ERROR:
    #         yield ResponseStatus.NA, ""
