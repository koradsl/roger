from __future__ import annotations

import logging
from typing import Optional, Any, Tuple, Type, Dict, Union, List

from roger.decorator import pack_to_chat_response, pack_to_embedding_response
from roger.memory.memory import Memory, MemoryCell
from roger.provider.oai.completion import OaiCompletion
from roger.provider.oai.embedding import OaiEmbedding
from roger.provider.status import ResponseStatus
from roger.response import RogerChatResponse

logger = logging.getLogger(__name__)


class Roger:
    __default_completion_model_name: str = "gpt-3.5-turbo-0613"
    __default_embedding_model_name: str = "text-embedding-ada-002"

    # Chat Components
    # - Memory
    __memory: Optional[Memory] = None

    # - Completion
    __completion: Optional[Type[OaiCompletion]] = None
    __completion_params: Optional[Dict[str, Union[str, int]]] = None

    __profile: Optional[MemoryCell] = None

    # Embedding Components
    __embedding: Optional[Type[OaiEmbedding]] = None
    __embedding_params: Optional[Dict[str, Union[str, int]]] = None

    def __init__(
        self,
        provider: str,
        profile: Optional[str] = None,
        completion_model_name: Optional[str] = None,
        completion_api_key: Optional[str] = None,
        completion_org_key: Optional[str] = None,
        completion_max_retries: int = 2,
        completion_timeout_ms: int = 20,
        memory_max_size: int = 7,
        memory_overflow_method: str = "fifo",
        embedding_model_name: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_org_key: Optional[str] = None,
        embedding_max_retries: int = 2,
        embedding_timeout_ms: int = 20,
    ):
        if completion_model_name:
            if provider == "oai":
                self.__completion = OaiCompletion
            else:
                raise ValueError("Unsupported provider")

            if completion_api_key is None and completion_org_key is None:
                raise ValueError("API key or organization key is None")

            __completion_model_name: Optional[str] = None
            __completion_keys: Optional[Tuple[str, str]] = None
            __completion_retry_options: Optional[Tuple[int, int]] = (2, 20)

            self.__completion_params = {
                "api_key": completion_api_key,
                "org_key": completion_org_key,
                "model_name": completion_model_name,
                "max_retries": completion_max_retries,
                "timeout_ms": completion_timeout_ms,
            }

            self.__memory = Memory(max_size=memory_max_size, overflow_method=memory_overflow_method)

            if profile:
                self.__profile = MemoryCell(role="system", content=profile)

        if embedding_model_name:
            if provider == "oai":
                self.__embedding = OaiEmbedding
            else:
                raise ValueError("Unsupported provider")

            self.__embedding_params = {
                "api_key": embedding_api_key,
                "org_key": embedding_org_key,
                "model_name": embedding_model_name,
                "max_retries": embedding_max_retries,
                "timeout_ms": embedding_timeout_ms,
            }

    @classmethod
    def with_chat(
        cls,
        provider: str,
        api_key: str,
        org_key: str,
        profile: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: int = 2,
        timeout_ms: int = 20,
        memory_max_size: int = 7,
        memory_overflow_method: str = "fifo",
    ) -> Roger:
        chat_roger = cls(
            provider=provider,
            completion_api_key=api_key,
            completion_org_key=org_key,
            profile=profile,
            completion_model_name=model_name if model_name else cls.__default_completion_model_name,
            completion_max_retries=max_retries,
            completion_timeout_ms=timeout_ms,
            memory_max_size=memory_max_size,
            memory_overflow_method=memory_overflow_method,
        )

        return chat_roger

    @classmethod
    def with_embedding(
        cls,
        provider: str,
        api_key: str,
        org_key: str,
        model_name: Optional[str] = None,
        max_retries: int = 2,
        timeout_ms: int = 20,
    ):
        embedding_roger = cls(
            provider=provider,
            embedding_api_key=api_key,
            embedding_org_key=org_key,
            embedding_model_name=model_name if model_name else cls.__default_embedding_model_name,
            embedding_max_retries=max_retries,
            embedding_timeout_ms=timeout_ms,
        )

        return embedding_roger

    def __to_inputs(self, content: str, **params: Any) -> Any:
        """입력 변환"""
        params["messages"] = [self.__profile.model_dump(mode="json")] if self.__profile else []
        params["messages"] += self.__memory.to_inputs()
        params["messages"] += [MemoryCell(role="user", content=content).model_dump(mode="json")]

        return params

    @pack_to_chat_response
    def call(self, content: str, **params: Any) -> Union[Tuple[ResponseStatus, str], RogerChatResponse]:
        logging.debug("hello!!!!")
        return self.__completion(**self.__completion_params).call(**self.__to_inputs(content, **params))

    @pack_to_chat_response
    async def acall(self, content: str, **params: Any) -> Union[Tuple[ResponseStatus, str], RogerChatResponse]:
        return await self.__completion(**self.__completion_params).acall(**self.__to_inputs(content, **params))

    @pack_to_embedding_response
    def embed(self, inputs: Union[List[str], str]) -> Tuple[ResponseStatus, List[Dict[str, Union[int, List[float]]]]]:
        return self.__embedding(**self.__embedding_params).call(inputs=inputs)
