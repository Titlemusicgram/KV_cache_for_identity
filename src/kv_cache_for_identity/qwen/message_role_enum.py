from enum import StrEnum, auto


class RoleEnum(StrEnum):
    USER = auto()
    SYSTEM = auto()
    ASSISTANT = auto()
