import asyncio
import time

from roger.response import RogerChatResponse, RogerEmbeddingResponse


def pack_to_chat_response(func):
    def runner(*args, **kwargs) -> RogerChatResponse:
        start_time = time.time()
        status, response = func(*args, **kwargs)
        end_time = time.time()

        return RogerChatResponse(status=status, content=response, collapsed_time=end_time - start_time)

    async def runner_async(*args, **kwargs) -> RogerChatResponse:
        start_time = time.time()
        status, response = await func(*args, **kwargs)
        end_time = time.time()

        return RogerChatResponse(status=status, content=response, collapsed_time=end_time - start_time)

    return runner if not asyncio.iscoroutinefunction(func) else runner_async


def pack_to_embedding_response(func):
    def runner(*args, **kwargs) -> RogerEmbeddingResponse:
        start_time = time.time()
        status, response = func(*args, **kwargs)
        end_time = time.time()

        return RogerEmbeddingResponse(status=status, features=response, collapsed_time=end_time - start_time)

    return runner
