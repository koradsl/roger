import os

import pytest

from roger.provider.oai.embedding import OaiEmbedding
from roger.provider.status import ResponseStatus

SAMPLE_TEXT = "Hello?"


@pytest.fixture()
def embedding():
    openai_api_key = os.getenv("TEST_OPENAI_API_KEY", "")
    openai_org_key = os.getenv("TEST_OPENAI_ORG_KEY", "")

    assert openai_api_key != ""
    assert openai_org_key != ""

    return OaiEmbedding(
        api_key=openai_api_key,
        org_key=openai_org_key,
    )


def test_get_embeddings(embedding):
    status, response = embedding.call(inputs=[SAMPLE_TEXT, SAMPLE_TEXT, SAMPLE_TEXT])

    assert type(status) == ResponseStatus
    assert isinstance(response, list)
    assert isinstance(response[0], dict)
    assert "idx" in response[0] and "features" in response[0]
    assert isinstance(response[0]["features"], list)
    assert isinstance(response[0]["features"][0], float)
    assert len(response[0]["features"]) == embedding.dimension


def test_get_embedding(embedding):
    status, response = embedding.call(inputs=SAMPLE_TEXT)

    assert type(status) == ResponseStatus
    assert isinstance(response, list)
    assert isinstance(response[0], dict)
    assert "idx" in response[0] and "features" in response[0]
    assert isinstance(response[0]["features"], list)
    assert isinstance(response[0]["features"][0], float)
    assert len(response[0]["features"]) == embedding.dimension
