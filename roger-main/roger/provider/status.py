from enum import IntEnum, auto


class ResponseStatus(IntEnum):
    SUCCESS = auto()  # Success to response
    WRONG_API_KEY_ERROR = auto()  # Wrong API Key
    PARAMETER_ERROR = auto()  # Necessary parameter is not given
    UNAVAILABLE_API_ERROR = auto()  # Unavailable API
    UNAVAILABLE_RESOURCE_ERROR = auto()  # Unavailable resource
    RATE_LIMIT_ERROR = auto()  # Rate Limit
    INTERNAL_SERVER_ERROR = auto()  # Retry or route another resource
    UNKNOWN_ERROR = auto()
