from typing import Callable, Union, Optional, Literal, List, Dict, Tuple

from openai import OpenAI
from roger.provider.oai.error import oai_error
from roger.provider.oai.keys import get_key_params
from roger.provider.status import ResponseStatus


class OaiEmbedding:
    __model_name: str = "text-embedding-ada-002"
    __token_size: int = 8191
    __dimension: int = 1536
    __encoding_format: Literal["float", "base64"] = "float"
    __create: Callable

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

        self.__create = OpenAI(**key_params).with_options(**self.__client_options).embeddings.create

    @property
    def token_size(self) -> int:
        return self.__token_size

    @property
    def dimension(self) -> int:
        return self.__dimension

    @oai_error
    def call(self, inputs: Union[List[str], str]) -> Tuple[ResponseStatus, List[Dict[str, Union[int, List[float]]]]]:
        """텍스트 임베딩"""

        if isinstance(inputs, str):
            inputs = [inputs]

        response = self.__create(model=self.__model_name, input=inputs, encoding_format=self.__encoding_format)

        return ResponseStatus.SUCCESS, [{"idx": data.index, "features": data.embedding} for data in response.data]
