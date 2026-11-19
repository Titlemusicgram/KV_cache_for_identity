from typing import Any
from dataclasses import dataclass, field
from torch import inference_mode, Tensor, softmax, tensor, long, cumsum, multinomial, topk
from transformers import Cache, DynamicCache, BatchEncoding, set_seed
from transformers.modeling_outputs import CausalLMOutputWithPast

from .qwen import DearQwen
from .qwen_helper_funcs import get_logits_n_cache_from_outputs
from .message_role_enum import RoleEnum


@dataclass
class StatefulQwenRuntime:
    qwen: DearQwen
    enable_thinking: bool
    max_new_tokens: int = 2000
    reset_seed_before_each_response: bool = False
    name: str = 'StatefulQwen'
    debug_token_ids: list[int] = field(default_factory=list[int])
    cache: Cache = field(init=False)
    eos_list: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.cache = DynamicCache(config=self.qwen.model.config)
        
        eos: list[int] | int = self.qwen.model.generation_config.eos_token_id # pyright: ignore[reportUnknownMemberType]
        if isinstance(eos, int):
            self.eos_list = [eos]
        elif isinstance(eos, list):
            self.eos_list = eos
        else:
            raise Exception('eos is not int or list[int]')

    def answer_the_prompt(self, prompt: str) -> str:
        if self.reset_seed_before_each_response:
            set_seed(1)
        logits = self.handle_user_message(prompt=prompt) # pyright: ignore[reportUnknownMemberType]
        answer = self.generate_answer_until_eos(logits)
        self.finalize_answer()
        return answer

    def select_next_token_id(self, logits: Tensor) -> int:
        if self.qwen.model.generation_config.temperature <= 0: # pyright: ignore[reportUnknownMemberType]
            raise ValueError("Temperature must be greater than 0")

        if not 0 < self.qwen.model.generation_config.top_p <= 1: # pyright: ignore[reportUnknownMemberType]
            raise ValueError("top_p must be in (0, 1]")

        last_token_logits: Tensor = logits[:, -1, :].float()

        scaled_logits = last_token_logits / self.qwen.model.generation_config.temperature # pyright: ignore[reportUnknownMemberType]
        probabilities = softmax(scaled_logits, dim=-1)

        sorted_values, sorted_indices = topk(input=probabilities, k=self.qwen.model.generation_config.top_k, dim=-1, sorted=True) # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        sorted_values = sorted_values / sorted_values.sum(dim=-1, keepdim=True)
        cumulative = cumsum(sorted_values, dim=-1)

        reached = cumulative >= self.qwen.model.generation_config.top_p # pyright: ignore[reportUnknownMemberType]
        cutoff_index = int(reached[0].nonzero()[0].item())
        candidate_count = cutoff_index + 1

        top_values = sorted_values[:, :candidate_count]
        top_indices = sorted_indices[:, :candidate_count]

        top_probs = top_values / top_values.sum(dim=-1, keepdim=True)
        selected_position = int(multinomial(top_probs, num_samples=1).item())

        next_token_id = int(top_indices[0][selected_position])

        return next_token_id

    def select_next_greedy_token_id(self, logits: Tensor) -> int:
        last_token_logits = logits[:, -1, :]
        next_token_id = int(last_token_logits.argmax(dim=-1).item())
        return next_token_id

    def decode_step(self, current_token: Tensor) -> Tensor:
        self.debug_token_ids.append(int(current_token[0, 0].item()))

        with inference_mode():
            next_outputs: CausalLMOutputWithPast = self.qwen.model(
                input_ids=current_token,
                past_key_values=self.cache,
                use_cache=True,
            )

        logits, cache = get_logits_n_cache_from_outputs(next_outputs)
        self.cache = cache
        return logits

    def append_to_cache(self, tokens: BatchEncoding[Any]) -> Tensor:
        input_ids = tokens["input_ids"][0].tolist()
        self.debug_token_ids.extend(input_ids)

        next_outputs: CausalLMOutputWithPast

        with inference_mode():
            next_outputs = self.qwen.model(
                input_ids=tokens["input_ids"],
                past_key_values=self.cache,
                use_cache=True,
            )

        logits, cache = get_logits_n_cache_from_outputs(next_outputs)
        self.cache = cache
        return logits

    def _handle_message(self, prompt: str, role: RoleEnum) -> Tensor:
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

        if isinstance(tokens, str) or isinstance(tokens, list):
            raise Exception("Wrong type of tokens")

        tokens = tokens.to(self.qwen.input_device)

        logits = self.append_to_cache(tokens=tokens)
        return logits

    def handle_internal_message(
        self, prompt: str, role: RoleEnum = RoleEnum.SYSTEM
    ) -> Tensor:
        return self._handle_message(prompt=prompt, role=role)

    def handle_user_message(
        self, prompt: str, role: RoleEnum = RoleEnum.USER
    ) -> Tensor:
        return self._handle_message(prompt=prompt, role=role)

    def generate_answer_until_eos(self, logits: Tensor) -> str:
        generated_token_ids: list[int] = []

        for _ in range(self.max_new_tokens):
            if self.qwen.model.generation_config.do_sample: # pyright: ignore[reportUnknownMemberType]
                next_token_id = self.select_next_token_id(logits=logits)
            else:
                next_token_id = self.select_next_greedy_token_id(logits=logits)

            next_token = tensor(
                data=[[next_token_id]],
                dtype=long,
                device=self.qwen.input_device,
            )

            if next_token_id in self.eos_list:
                self.decode_step(current_token=next_token)
                break

            logits = self.decode_step(current_token=next_token)
            generated_token_ids.append(next_token_id)

        answer = self.qwen.tokenizer.decode(generated_token_ids)  # pyright: ignore[reportUnknownMemberType]
        if not isinstance(answer, str):
            raise Exception("Qwen's answer was list[str]. Should be just a str. Check out qwen_runtime.generate_answer()")

        return answer

    def append_chat_separator(self) -> None:
        tokens = self.qwen.tokenizer(
            text="\n",
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokens = tokens.to(self.qwen.input_device)
        self.append_to_cache(tokens=tokens)

    def finalize_answer(self) -> None:
        self.append_chat_separator()

    def generate_internal_response(self, prompt: str) -> str:
        logits = self.handle_internal_message(prompt=prompt)
        answer = self.generate_answer_until_eos(logits=logits)
        self.finalize_answer()
        return answer

    def show_tokens(self) -> None:
        print(self.debug_token_ids)
