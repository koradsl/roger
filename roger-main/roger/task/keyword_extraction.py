from typing import Tuple, List

from roger import ResponseStatus
from roger.task.base import BaseTaskMixin

PROFILE = """Your task is to extract important and semantic keywords among given document"""

INPUT_FORMAT: str = """
D : {DOCUMENT}

Questions must be separated by &&. For example, K : {KEYWORD1} && {KEYWORD2} && ...
K : """


class KeywordExtractionTask(BaseTaskMixin):
    def __new__(
        cls,
        text: str,
        api_key: str,
        org_key: str,
    ) -> Tuple[ResponseStatus, List[str]]:
        inputs = cls.formatter(
            text=INPUT_FORMAT,
            replacements=[
                ("{DOCUMENT}", text),
            ],
        )

        response = cls.completion(text=inputs, api_key=api_key, org_key=org_key, profile=PROFILE)

        if response.status != ResponseStatus.SUCCESS:
            return response.status, []

        return response.status, response.content.split(" && ")
