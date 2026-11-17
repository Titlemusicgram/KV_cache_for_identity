from dataclasses import dataclass, field
from transformers import BatchEncoding, set_seed
from torch import Tensor, tensor, long, ones_like

from .qwen import DearQwen
from .message_role_enum import RoleEnum


@dataclass
class StatelessQwenRuntime:
    qwen: DearQwen
    enable_thinking: bool
    max_new_tokens: int = 2000
    reset_seed_before_each_response: bool = False
    name: str = 'StatelessQwen'
    token_history: list[int] = field(default_factory=list[int])
    debug_token_ids: list[int] = field(default_factory=list[int])
    
    def answer_the_prompt(self, prompt: str) -> str:
        if self.reset_seed_before_each_response:
            set_seed(1)
        return self.generate_with_token_history(prompt=prompt, role = RoleEnum.USER)

    def generate_with_token_history(self, prompt: str, role: RoleEnum) -> str:
        self.debug_token_ids = self.token_history

        messages: list[dict[str, str]] = [
            {
                "role": role,
                "content": prompt,
            }
        ]

        tokens = self.qwen.tokenizer.apply_chat_template(  # pyright: ignore[reportUnknownMemberType]
            conversation=messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=self.enable_thinking,
        )
        if not isinstance(tokens, BatchEncoding):
            raise TypeError("tokens is not a BatchEncoding")

        input_ids = tokens["input_ids"][0].tolist()
        self.token_history.extend(input_ids)
        input_length = len(self.token_history)

        input_ids = tensor(
            [self.token_history],
            dtype=long,
            device=self.qwen.model.device,
        )
        attention_mask = ones_like(input_ids)

        outputs = self.qwen.model.generate(  # pyright: ignore[reportUnknownMemberType]
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
        )

        generated_ids: Tensor = outputs[0][input_length:]

        if not isinstance(generated_ids, Tensor):
            raise TypeError("answer is not a Tensor")

        generated_ids_list: list[int] = (
            generated_ids.tolist()
        )  # pyright: ignore[reportUnknownMemberType]

        self.token_history.extend(generated_ids_list)
        self.append_chat_separator()

        answer = self.qwen.tokenizer.decode(  # pyright: ignore[reportUnknownMemberType]
            token_ids=generated_ids,
            skip_special_tokens=True,
        )
        if not isinstance(answer, str):
            raise TypeError("answer is not a str")

        return answer

    def generate_internal_response(self, prompt: str):
        return self.generate_with_token_history(prompt=prompt, role=RoleEnum.SYSTEM)

    def show_tokens(self) -> None:
        print(self.debug_token_ids)

    def append_chat_separator(self) -> None:
        tokens = self.qwen.tokenizer(
            text="\n",
            add_special_tokens=False,
        )
        separator_ids = tokens["input_ids"]
        self.token_history.extend(separator_ids)
