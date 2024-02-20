import asyncio
import logging
from typing import Tuple, Optional, Union, Iterator, AsyncIterator

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    RetryCallState,
    before_sleep_log,
)

from roger.provider.status import ResponseStatus

"""
# Status Code : 400, 요구된 필수 파라미터를 입력하지 않았을 경우 -> openai.BadRequestError
# Status Code : 401, API 키가 잘못 되었거나, 만료되었을 경우 -> openai.AuthenticationError
# Status Code : 403, API 키, Organization 키 또는 리소스 키가 잘못 되었을 경우 -> openai.PermissionDeniedError
# Status Code : 404, 요청한 리소스가 존재하지 않은 경우 -> openai.NotFoundError
# Status Code : N/A, OpenAI 서버와의 연결이 불안정한 경우 -> openai.APIConnectionError
# Status Code : 422, 입력은 잘 들어갔지만 요청이 처리되지 않은 경우 -> openai.UnprocessableEntityError
# Status Code : 429, Rate Limit을 초과한 경우 -> openai.RateLimitError
# Status Code : >= 500, OpenAI 서버가 불안정한 경우 -> openai.InternalServerError
"""

"""
JITTER Algorithm 

Equation : min(__RETRY_INITIAL * 2 ** n + random.uniform(0, __RETRY_JITTER), __RETRY_MAXIMUM)
"""
__RETRY_INITIAL = 1
__RETRY_JITTER = 5.0
__RETRY_MAXIMUM = 60
__RETRY_ATTEMPT = 3

__OAI_CRITICAL_EXCEPTIONS = (
    # Status Code : 400
    # 요구된 필수 파라미터를 입력하지 않았을 경우. Max token error도 여기에 포함됨
    openai.BadRequestError,
    # Status Code : 401
    # - API 키가 잘못 되었거나, 만료되었을 경우
    openai.AuthenticationError,
    # Status Code : 403
    # API 키, Organization 키 또는 리소스 키가 잘못 되었을 경우
    openai.PermissionDeniedError,
    # Status Code : 404
    # 요청한 리소스가 존재하지 않은 경우
    openai.NotFoundError,
)


__OAI_RETRY_EXCEPTIONS = (
    # Status Code : N/A
    # OpenAI 서버와의 연결이 불안정한 경우
    openai.APIConnectionError,
    # Status Code : 422
    # 입력은 잘 들어갔지만 요청이 처리되지 않은 경우
    openai.UnprocessableEntityError,
    # Status Code : 429
    # Rate Limit을 초과한 경우
    openai.RateLimitError,
    # Status Code : >= 500
    # OpenAI 서버가 불안정한 경우
    openai.InternalServerError,
)


def on_retry_error_callback(state: RetryCallState) -> Tuple[ResponseStatus, Optional[str]]:
    error = state.outcome._exception  # type: ignore

    if isinstance(error, openai.APIConnectionError):
        return ResponseStatus.INTERNAL_SERVER_ERROR, None

    if isinstance(error, openai.UnprocessableEntityError):
        return ResponseStatus.INTERNAL_SERVER_ERROR, None

    if isinstance(error, openai.RateLimitError):
        return ResponseStatus.RATE_LIMIT_ERROR, None

    if isinstance(error, openai.InternalServerError):
        return ResponseStatus.UNAVAILABLE_API_ERROR, None

    return ResponseStatus.UNKNOWN_ERROR, None


def oai_error(f):
    @retry(
        retry=retry_if_exception_type(exception_types=__OAI_RETRY_EXCEPTIONS),
        wait=wait_exponential_jitter(initial=__RETRY_INITIAL, max=__RETRY_MAXIMUM, jitter=__RETRY_JITTER),
        stop=stop_after_attempt(__RETRY_ATTEMPT),
        retry_error_callback=on_retry_error_callback,
        before_sleep=before_sleep_log(logging.getLogger(), logging.DEBUG),
    )
    def wrapper(
        *args, **kwargs
    ) -> Union[Tuple[ResponseStatus, Optional[str]], Iterator[Tuple[ResponseStatus, Optional[str]]]]:
        try:
            return f(*args, **kwargs)
        except __OAI_CRITICAL_EXCEPTIONS as e:
            raise e

    @retry(
        retry=retry_if_exception_type(exception_types=__OAI_RETRY_EXCEPTIONS),
        wait=wait_exponential_jitter(initial=__RETRY_INITIAL, max=__RETRY_MAXIMUM, jitter=__RETRY_JITTER),
        stop=stop_after_attempt(__RETRY_ATTEMPT),
        retry_error_callback=on_retry_error_callback,
        before_sleep=before_sleep_log(logging.getLogger(), logging.DEBUG),
    )
    async def wrapper_async(
        *args, **kwargs
    ) -> Union[Tuple[ResponseStatus, Optional[str]], AsyncIterator[Tuple[ResponseStatus, Optional[str]]]]:
        try:
            return await f(*args, **kwargs)
        except __OAI_CRITICAL_EXCEPTIONS as e:
            raise e

    return wrapper if not asyncio.iscoroutinefunction(f) else wrapper_async
