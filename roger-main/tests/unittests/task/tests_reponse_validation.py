from roger.task.response_validation import ResponseValidationTask

from tests.sample import openai_keys

SAMPLE_PROFILE = """Your task is to classify a user query among given labels"""

SAMPLE_QUERY = """이 문서에 대해 설명해줘

Label list
1. Summarization : A user want to summarize about some documents
2. Question and Answering : A user want to receive an answer about question
3. None : If this is a query that does not belong to the label above


The output must be a single number in label list number. For example, A : 1
A : """

SAMPLE_ANSWER = """3"""


def test_response_validation_task(openai_keys):
    openai_api_key, openai_org_key = openai_keys

    status, is_valid = ResponseValidationTask(
        question=SAMPLE_QUERY,
        answer=SAMPLE_ANSWER,
        profile=SAMPLE_PROFILE,
        api_key=openai_api_key,
        org_key=openai_org_key,
    )

    assert isinstance(is_valid, bool)

    print(is_valid)
