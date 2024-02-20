from roger.task.summarization import SummarizationTask
from tests.sample import SHORT_TEXT, LONG_TEXT, openai_keys


def test_call_summarization_about_short_text(openai_keys):
    openai_api_key, openai_org_key = openai_keys

    _, _, text = SummarizationTask.call(text=SHORT_TEXT, api_key=openai_api_key, org_key=openai_org_key)

    assert len(text) < len(SHORT_TEXT)

    print(text)


def test_call_summarization_about_long_text(openai_keys):
    openai_api_key, openai_org_key = openai_keys

    _, _, text = SummarizationTask.call(text=LONG_TEXT, api_key=openai_api_key, org_key=openai_org_key)

    assert len(text) < len(LONG_TEXT)

    print(text)


def test_stream_summarization_about_short_text(openai_keys):
    text = ""
    openai_api_key, openai_org_key = openai_keys

    _, _, response = SummarizationTask.call(text=SHORT_TEXT, api_key=openai_api_key, org_key=openai_org_key)

    for chunk in response:
        text += chunk

    assert len(text) < len(SHORT_TEXT)

    print(text)


def test_stream_summarization_about_long_text(openai_keys):
    text = ""
    openai_api_key, openai_org_key = openai_keys

    _, _, response = SummarizationTask.call(text=LONG_TEXT, api_key=openai_api_key, org_key=openai_org_key)

    for chunk in response:
        text += chunk

    assert len(text) < len(SHORT_TEXT)

    print(text)
