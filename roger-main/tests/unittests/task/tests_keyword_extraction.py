from roger.task.keyword_extraction import KeywordExtractionTask
from tests.sample import openai_keys, LONG_TEXT


def test_keyword_extraction_task(openai_keys):
    openai_api_key, openai_org_key = openai_keys

    _, keywords = KeywordExtractionTask(text=LONG_TEXT, api_key=openai_api_key, org_key=openai_org_key)
    assert isinstance(keywords, list)

    print(keywords)
