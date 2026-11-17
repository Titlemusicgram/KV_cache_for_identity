from dataclasses import dataclass


@dataclass(frozen=True)
class DialogExperimentConditions:
    author: str
    name: str
    enable_thinking: bool
    intro_prompt: str
    first_speaker_prompt: str
    final_prompt: str
    reset_seed_before_each_response: bool
    rounds: int
    stateless_qwen_first: bool
