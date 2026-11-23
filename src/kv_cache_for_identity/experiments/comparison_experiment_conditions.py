from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonExperimentConditions:
    type: str
    author: str
    name: str
    enable_thinking: bool
    prompts: list[str]
    reset_seed_before_each_response: bool
    seed: int
    max_new_tokens: int
