import os
from typing import Tuple

import openai
import pytest

from roger.provider.oai.completion import OaiCompletion

SAMPLE_TEXT = "Hello?"


@pytest.fixture()
def keys() -> Tuple[str, str]:
    openai_api_key = os.getenv("TEST_OPENAI_API_KEY", "")
    openai_org_key = os.getenv("TEST_OPENAI_ORG_KEY", "")

    assert openai_api_key != ""
    assert openai_org_key != ""

    return openai_api_key, openai_org_key



def test_authentication_error(keys):
    """ API Key가 잘못된 경우 -> openai.AuthenticationError 발생 시도 """
    openai_api_key, openai_org_key = keys

    completion = OaiCompletion(
        model_name="gpt-3.5-turbo-0613",
        api_key=openai_api_key + "1",
        org_key=openai_org_key,
    )

    with pytest.raises(openai.AuthenticationError):
        completion.call(
            messages=[{"role": "user", "content": SAMPLE_TEXT}],
            temperature=1.0,
            max_tokens=256,
        )


def test_permission_denied_error(keys):
    """ Organization key가 잘못된 경우 -> openai.PermissionDeniedError 발생 시도 """
    # TODO : 구현 예정
    raise NotImplementedError()


def test_exceed_max_tokens(keys):
    """ Token 크기가 너무 큰 경우 -> openai.AuthenticationError 발생 시도 """
    openai_api_key, openai_org_key = keys

    completion = OaiCompletion(
        model_name="gpt-3.5-turbo-0613",
        api_key=openai_api_key,
        org_key=openai_org_key,
    )

    with pytest.raises(openai.BadRequestError):
        completion.call(
            messages=[{"role": "user", "content": SAMPLE_TEXT * 10000}],
            temperature=1.0,
            max_tokens=256,
        )