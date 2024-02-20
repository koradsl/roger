import asyncio
import os
from typing import Tuple

import pytest

from roger.provider.oai.completion import OaiCompletion
from roger.provider.status import ResponseStatus

SAMPLE_TEXT = "Hello?"


@pytest.fixture()
def keys() -> Tuple[str, str]:
    openai_api_key = os.getenv("TEST_OPENAI_API_KEY", "")
    openai_org_key = os.getenv("TEST_OPENAI_ORG_KEY", "")

    assert openai_api_key != ""
    assert openai_org_key != ""

    return openai_api_key, openai_org_key


def test_call_oai(keys):
    openai_api_key, openai_org_key = keys

    completion = OaiCompletion(
        model_name="gpt-3.5-turbo-0613",
        api_key=openai_api_key,
        org_key=openai_org_key,
    )

    status, response = completion.call(
        messages=[{"role": "user", "content": SAMPLE_TEXT}],
        temperature=1.0,
        max_tokens=256,
    )

    assert type(status) == ResponseStatus
    assert response != ""

    print(status, response)



def test_acall_oai(keys):
    openai_api_key, openai_org_key = keys

    completion = OaiCompletion(
        model_name="gpt-3.5-turbo-0613",
        api_key=openai_api_key,
        org_key=openai_org_key,
    )

    loop = asyncio.get_event_loop()
    status, response = loop.run_until_complete(
        completion.acall(
            messages=[{"role": "user", "content": SAMPLE_TEXT}],
            temperature=1.0,
            max_tokens=256,
        )
    )
    loop.close()

    assert type(status) == ResponseStatus
    assert response != ""

    print(response)


def test_stream_oai(completion):
    openai_api_key, openai_org_key = keys

    completion = OaiCompletion(
        model_name="gpt-3.5-turbo-0613",
        api_key=openai_api_key,
        org_key=openai_org_key,
    )

    stream = completion.stream(
        messages=[{"role": "user", "content": SAMPLE_TEXT}],
        temperature=1.0,
        max_tokens=256,
    )

    for status, chunk in stream:
        assert type(status) == ResponseStatus
        assert isinstance(chunk, str)

        print(chunk, end="")


def test_astream_openai(completion):
    # loop = asyncio.get_event_loop()
    # stream = loop.run_until_complete(
    #     completion.astream(
    #         messages=[{"role": "user", "content": SAMPLE_TEXT}],
    #         temperature=1.0,
    #         max_tokens=256,
    #     )
    # )
    # loop.close()
    # TODO : Async Streaming 테스트 코드 작성
    raise NotImplementedError()


def test_stress_call_oai(keys):
    openai_api_key, openai_org_key = keys

    completion = OaiCompletion(
        # model_name="gpt-3.5-turbo-0613",
        model_name="gpt-4-0613",
        api_key=openai_api_key,
        org_key=openai_org_key,
    )

    status, response = completion.call(
        messages=[{"role": "user", "content": SAMPLE_TEXT}],
        temperature=1.0,
        max_tokens=256,
    )

    assert type(status) == ResponseStatus
    assert response != ""

    print(status, response)

def test_stress_acall_oai(keys):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    openai_api_key, openai_org_key = keys

    completion = OaiCompletion(
        model_name="gpt-3.5-turbo-0613",
        # model_name="gpt-4-0613",
        api_key=openai_api_key,
        org_key=openai_org_key,
    )


    async def acall():
        futures = [asyncio.ensure_future(completion.acall(
                messages=[{"role": "user", "content": SAMPLE_TEXT}],
                temperature=1.0,
                max_tokens=256,
            )) for _ in range(8)]
        results = await asyncio.gather(*futures)

        return results

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(acall())
    loop.close()

    for status, response in results:
        print(status, response)

    import time
    time.sleep(1)
