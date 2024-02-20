from typing import Tuple, List

from roger import ResponseStatus
from roger.task.base import BaseTaskMixin

PROFILE = """Your task is to generation next semantic similar question when given user query and bot answer"""

SEPERATOR = "<SEP>"

INPUT_FORMAT: str = """
Q : {QUERY}
A : {ANSWER}

Please create up to 3 questions. Each question must be up to 50 words. 
Questions must be separated by {SEPERATOR}. 
For example, L : {QUESTION1} {SEPERATOR} {QUESTION2} {SEPERATOR} {QUESTION3}

L : """


class QuestionGenerationTask(BaseTaskMixin):
    def __new__(
        cls,
        question: str,
        answer: str,
        api_key: str,
        org_key: str,
    ) -> Tuple[ResponseStatus, List[str]]:
        inputs = cls.formatter(
            text=INPUT_FORMAT,
            replacements=[
                ("{QUERY}", question),
                ("{ANSWER}", answer),
                ("{SEPERATOR}", SEPERATOR),
            ],
        )

        response = cls.completion(text=inputs, api_key=api_key, org_key=org_key, profile=PROFILE)

        if response.status != ResponseStatus.SUCCESS:
            return response.status, []

        return response.status, response.content.split(SEPERATOR)
