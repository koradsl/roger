from typing import Optional, Dict, Tuple

from roger import ResponseStatus
from roger.task.base import BaseTaskMixin

DEFAULT_LABELS: Dict[str, str] = {
    "Summarization": "A user want to summarize about some documents",
    "Question and Answering": "A user want to receive an answer about question",
    "None": "If this is a query that does not belong to the label above",
}

PROFILE = """Your task is to classify a user query among given labels"""

INPUT_FORMAT: str = """
Q : {QUERY}

Label list
{LABELS}

The output must be a single number in label list number. For example, A : 1
A : """


class QueryClassificationTask(BaseTaskMixin):
    def __new__(
        cls,
        query: str,
        api_key: str,
        org_key: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Tuple[ResponseStatus, int]:
        label_list = ""

        if not labels:
            labels = DEFAULT_LABELS

        for idx, (label, description) in enumerate(labels.items()):
            label_list += f"{str(int(idx) + 1)}. {label} : {description}\n"

        inputs = cls.formatter(
            text=INPUT_FORMAT,
            replacements=[
                ("{QUERY}", query),
                ("{LABELS}", label_list),
            ],
        )

        response = cls.completion(text=inputs, api_key=api_key, org_key=org_key, profile=PROFILE)

        if response.status != ResponseStatus.SUCCESS:
            return response.status, -1

        label = response.content

        if not label.isdigit():
            return response.status, -1

        return response.status, int(label)
