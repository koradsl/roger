import asyncio
import os

import pytest

from roger import Roger, RogerChatResponse, RogerEmbeddingResponse

PROFILE = "Your are very smart agent"
SAMPLE_TEXT = "Hello?"
MEMORY_MAX_SIZE = 7
MEMORY_OVERFLOW_METHOD = "fifo"


@pytest.fixture()
def chat_roger() -> Roger:
    openai_api_key = os.getenv("TEST_OPENAI_API_KEY", "")
    openai_org_key = os.getenv("TEST_OPENAI_ORG_KEY", "")

    assert openai_api_key != ""
    assert openai_org_key != ""

    return Roger.with_chat(
        provider="oai",
        api_key=openai_api_key,
        org_key=openai_org_key,
        memory_max_size=MEMORY_MAX_SIZE,
        memory_overflow_method=MEMORY_OVERFLOW_METHOD,
    )


@pytest.fixture()
def embedding_roger() -> Roger:
    openai_api_key = os.getenv("TEST_OPENAI_API_KEY", "")
    openai_org_key = os.getenv("TEST_OPENAI_ORG_KEY", "")

    assert openai_api_key != ""
    assert openai_org_key != ""

    return Roger.with_embedding(
        provider="oai",
        api_key=openai_api_key,
        org_key=openai_org_key,
    )


def test_call(chat_roger):
    response = chat_roger.call(content=SAMPLE_TEXT)

    assert isinstance(response, RogerChatResponse)

    print(response)


def test_acall(chat_roger):
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(chat_roger.acall(content=SAMPLE_TEXT))
    loop.close()

    assert isinstance(response, RogerChatResponse)

    print(response)


def test_embedding(embedding_roger):
    response = embedding_roger.embed(inputs=[SAMPLE_TEXT, SAMPLE_TEXT])

    assert isinstance(response, RogerEmbeddingResponse)

    print(response)


def testss():
    import os

    print(os.getenv("TESTT"))
