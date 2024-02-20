from roger.task.base import BaseTaskMixin
from roger import ResponseStatus
from typing import Tuple


PROFILE: str = """
Your task is to make sure that the bot answer correctly to user query, depending on the profile of the bot.
"""


INPUT_FORMAT: str = """
Bot profile

{BOT_PROFILE}
-----
User query

{QUERY}
-----
Bot answer

{BOT_ANSWER}
-----

The output must be a True or False. For example, V : True

V : """


class ResponseValidationTask(BaseTaskMixin):
    def __new__(
        cls,
        question: str,
        answer: str,
        profile: str,
        api_key: str,
        org_key: str,
    ) -> Tuple[ResponseStatus, bool]:
        inputs = cls.formatter(
            text=INPUT_FORMAT,
            replacements=[
                ("{BOT_PROFILE}", profile),
                ("{QUERY}", question),
                ("{BOT_ANSWER}", answer),
            ],
        )

        response = cls.completion(text=inputs, api_key=api_key, org_key=org_key, profile=PROFILE)

        if response.status != ResponseStatus.SUCCESS:
            return response.status, False

        if "True" not in response.content:
            return response.status, False

        return response.status, True
