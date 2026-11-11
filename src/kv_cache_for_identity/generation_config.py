from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    do_sample: bool
    enable_thinking: bool
    temperature: float
    top_p: float
        