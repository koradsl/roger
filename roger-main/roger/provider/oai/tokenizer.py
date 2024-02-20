import tiktoken
from typing import Dict, List


class OaiTokenizer:
    @classmethod
    def count_tokens(cls, text: str, model_name: str = "cl100k_base") -> int:
        """단일 텍스트 토큰 개수 체크"""
        encoder = tiktoken.get_encoding(encoding_name=model_name)

        return len(encoder.encode(text=text))

    @classmethod
    def count_tokens_from_messages(cls, messages: List[Dict[str, str]], model_name: str = "cl100k_base") -> int:
        """메시지 토큰 개수 체크"""

        """
        reference
        - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """

        encoder = tiktoken.get_encoding(encoding_name=model_name)
        num_tokens = 0
        tokens_per_message = 3
        tokens_per_name = 1

        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoder.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        return num_tokens
