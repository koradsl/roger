from collections import deque
from typing import Optional, Deque, List, Dict

from roger.memory.cell import MemoryCell
from roger.memory.overflow import MemoryOverflowHandler


class Memory:
    register: Deque[MemoryCell] = deque([])

    # Default memory overflow handler uses FIFO method
    overflow_handler = MemoryOverflowHandler.fifo

    def __init__(self, max_size: int, overflow_method: str = "fifo"):
        self.max_size = max_size

        if overflow_method != "fifo":
            raise ValueError("Unsupported overflow method")

    @property
    def size(self) -> int:
        """저장된 메모리 셀 개수"""
        return len(self.register)

    def is_empty(self) -> bool:
        """저장된 메모리 셀이 없는지 확인"""
        return len(self.register) == 0

    def is_full(self) -> bool:
        """버퍼에 메모리 셀이 모두 차있는지 확인"""
        return len(self.register) == self.max_size

    def to_inputs(self) -> List[Dict[str, str]]:
        """메모리 셀들을 JSON 포맷으로 변환"""
        return [reg.model_dump(mode="json") for reg in self.register]

    def push(self, role: str, content: str) -> None:
        """메모리 셀 추가"""
        if self.is_full():
            self.overflow_handler(register=self.register)

        self.register.append(MemoryCell(role=role, content=content))

    def pop(self, index: int = -1) -> Optional[MemoryCell]:
        """특정 index에 대한 메모리 셀 제거 및 반환"""
        if self.is_empty():
            return None

        cell = self.register[index]
        del self.register[index]

        return cell
