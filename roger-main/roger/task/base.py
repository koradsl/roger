from roger import Roger, RogerChatResponse
from typing import Tuple, List, Union, Optional


class BaseTaskMixin:
    @staticmethod
    def formatter(text: str, replacements: List[Tuple[str, str]]) -> str:
        for replacement in replacements:
            text = text.replace(replacement[0], replacement[1])

        return text

    @staticmethod
    def completion(text: str, api_key: str, org_key: str, profile: Optional[str] = None) -> RogerChatResponse:
        roger = Roger.with_chat(provider="oai", api_key=api_key, org_key=org_key, profile=profile)

        return roger.call(content=text)

    @staticmethod
    def embedding(texts: Union[List[str], str], api_key: str, org_key: str):
        roger = Roger.with_embedding(provider="oai", api_key=api_key, org_key=org_key)

        return roger.embed(inputs=texts)
