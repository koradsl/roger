from typing import Deque
from roger.memory.cell import MemoryCell


class MemoryOverflowHandler:
    """메모리 Overflow 발생 시, 동작함"""

    @classmethod
    def fifo(cls, register: Deque[MemoryCell]) -> None:
        """FIFO 방식"""
        register.popleft()
