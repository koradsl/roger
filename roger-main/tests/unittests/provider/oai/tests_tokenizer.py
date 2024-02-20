import asyncio
import os
from typing import Generator

import pytest

from roger.provider.oai.tokenizer import OpenAIClient, OpenAITokenizer, OpenAIEmbedding

SAMPLE_TEXT = "Hello?"


@pytest.fixture()
def openai():
    openai_api_key = os.getenv("TEST_OPENAI_API_KEY", "")
    openai_org_key = os.getenv("TEST_OPENAI_ORG_KEY", "")

    assert openai_api_key != ""
    assert openai_org_key != ""

    return OpenAIClient(
        model_name="gpt-3.5-turbo-0613",
        api_key=openai_api_key,
        org_key=openai_org_key,
    )


@pytest.fixture()
def openai_embedding():
    openai_api_key = os.getenv("TEST_OPENAI_API_KEY", "")
    openai_org_key = os.getenv("TEST_OPENAI_ORG_KEY", "")

    assert openai_api_key != ""
    assert openai_org_key != ""

    return OpenAIEmbedding(
        api_key=openai_api_key,
        org_key=openai_org_key,
    )


def test_call_openai(openai):
    response = openai.call(
        messages=[{"role": "user", "content": SAMPLE_TEXT}],
        temperature=1.0,
        max_tokens=256,
    )

    assert response != ""

    print(response)


def test_acall_openai(openai):
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(
        openai.acall(
            messages=[{"role": "user", "content": SAMPLE_TEXT}],
            temperature=1.0,
            max_tokens=256,
        )
    )
    loop.close()

    assert response != ""

    print(response)


def test_stream_openai(openai):
    response = openai.stream(
        messages=[{"role": "user", "content": SAMPLE_TEXT}],
        temperature=1.0,
        max_tokens=256,
    )

    assert isinstance(response, Generator)

    for chunk in response:
        print(chunk, end="")


def test_astream_openai(openai):
    # TODO : Test Case 작성
    ...


def test_count_tokens():
    len_tokens = OpenAITokenizer.count_tokens(text="홍길동")

    assert len_tokens == 5


def test_count_tokens_from_messages():
    messages = [{"role": "user", "content": SAMPLE_TEXT}]

    len_message_tokens = OpenAITokenizer.count_tokens_from_messages(messages=messages)

    assert len_message_tokens != 0

    print(len_message_tokens)


def test_get_embeddings(openai_embedding):
    features = openai_embedding.call(texts=[SAMPLE_TEXT, SAMPLE_TEXT, SAMPLE_TEXT])

    assert isinstance(features, list)
    assert isinstance(features[0], list)
    assert isinstance(features[0][0], float)
    assert len(features[0]) == openai_embedding.dimension


def test_get_embedding(openai_embedding):
    features = openai_embedding.call(texts=SAMPLE_TEXT)

    assert isinstance(features, list)
    assert isinstance(features[0], float)
    assert len(features) == openai_embedding.dimension
