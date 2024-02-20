from pydantic import BaseModel, field_validator


class MemoryCell(BaseModel):
    """메모리 셀"""

    role: str
    content: str

    @classmethod
    @field_validator("role")
    def valid_type(cls, role: str):
        """role 타입 체크"""
        assert role in ["system", "assistant", "user"]

        return role
