from typing import Protocol


class QwenRuntime(Protocol):
    name: str
    reset_seed_before_each_response: bool

    def answer_the_prompt(self, prompt: str) -> str: ...
    def show_tokens(self) -> None: ...
