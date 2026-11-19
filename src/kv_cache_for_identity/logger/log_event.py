from dataclasses import dataclass


@dataclass(frozen=True)
class LogEvent:
    type: str
    step: int
    speaker: str
    receiver: str
    input: str
    raw_output: str
    public_output: str
