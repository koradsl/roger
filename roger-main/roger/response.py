from typing import List, Dict, Union, Optional

from pydantic import BaseModel, field_validator

from roger.provider.status import ResponseStatus


class RogerBaseResponse(BaseModel):
    status: ResponseStatus
    collapsed_time: float  # Unit : second

    @classmethod
    @field_validator("collapsed_time")
    def round_collapsed_time(cls, collapsed_time: float):
        return round(collapsed_time, 4)


class RogerChatResponse(RogerBaseResponse):
    content: Optional[str]


class RogerEmbeddingResponse(RogerBaseResponse):
    features: List[Dict[str, Union[int, List[float]]]]
