from dataclasses import dataclass, field
from transformers import BatchEncoding
from torch import Tensor, tensor, long, ones_like, topk

from .qwen import DearQwen
from .message_role_enum import RoleEnum

from ..generation_config import GenerationConfig


@dataclass
class StatelessQwenRuntime:
    qwen: DearQwen
    generation_config: GenerationConfig
    messages: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    token_history: list[int] = field(default_factory=list[int])
    debug_token_ids: list[int] = field(default_factory=list[int])

    def generate_with_token_history(self, prompt: str, role: RoleEnum) -> str:
        self.debug_token_ids = self.token_history

        messages: list[dict[str, str]] = [{
            "role": role,
            "content": prompt,
        }]
        
        tokens = self.qwen.tokenizer.apply_chat_template( # pyright: ignore[reportUnknownMemberType]
            conversation=messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt',
            return_dict=True,
            enable_thinking=self.generation_config.enable_thinking,
        )
        if not isinstance(tokens, BatchEncoding):
            raise TypeError('tokens is not a BatchEncoding')

        input_ids = tokens['input_ids'][0].tolist()
        self.token_history.extend(input_ids)
        input_length = len(self.token_history)

        input_ids = tensor(
            [self.token_history],
            dtype=long,
            device=self.qwen.model.device,
        )
        attention_mask = ones_like(input_ids)
        
        outputs = self.qwen.model.generate( # pyright: ignore[reportUnknownMemberType]
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.generation_config.max_new_tokens,
            do_sample=self.generation_config.do_sample,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            return_dict_in_generate=True,
            output_logits=True,
        )

        # generated_ids: Tensor = outputs[0][input_length:]
        generated_ids: Tensor = outputs.sequences[0][input_length:]

        if not isinstance(generated_ids, Tensor):
            raise TypeError('answer is not a Tensor')
            
        generated_ids_list: list[int] = generated_ids.tolist() # pyright: ignore[reportUnknownMemberType]

        self.token_history.extend(generated_ids_list)
        self.append_chat_separator()

        answer = self.qwen.tokenizer.decode( # pyright: ignore[reportUnknownMemberType]
            token_ids=generated_ids,
            skip_special_tokens=False,
        )
        if not isinstance(answer, str):
            raise TypeError('answer is not a str')

        target_step = 738 - input_length
        if 0 <= target_step < len(outputs.logits):
            target_logits = outputs.logits[target_step][0]
            top_values, top_indices = topk(target_logits, k=5)
            for top_value, top_index in zip(top_values, top_indices):
                token_id = int(top_index.item())
                logit_value = float(top_value.item())
                print(f'{token_id} | {self.qwen.tokenizer.decode(token_id)} | {logit_value}')
                                
        return answer

    # def generate_with_message_history(self, prompt: str, role: RoleEnum) -> str:
    #     self.messages.append({
    #         "role": role,
    #         "content": prompt,
    #     })
        
    #     tokens = self.qwen.tokenizer.apply_chat_template( # pyright: ignore[reportUnknownMemberType]
    #         conversation=self.messages,
    #         tokenize=True,
    #         add_generation_prompt=True,
    #         return_tensors='pt',
    #     )
    #     if not isinstance(tokens, BatchEncoding):
    #         raise TypeError('tokens is not a BatchEncoding')

    #     input_ids = tokens['input_ids'][0].tolist()
    #     self.debug_token_ids = input_ids

    #     tokens = tokens.to(self.qwen.model.device)
                
    #     outputs = self.qwen.model.generate( # pyright: ignore[reportUnknownMemberType]
    #         **tokens,
    #         max_new_tokens=self.generation_config.max_new_tokens,
    #         do_sample=self.generation_config.do_sample,
    #         temperature=self.generation_config.temperature,
    #         top_p=self.generation_config.top_p,
    #         return_dict_in_generate=True,
    #         output_logits=True,
    #     )

    #     input_length = tokens["input_ids"].shape[-1]
    #     generated_ids: Tensor = outputs[0][input_length:]

    #     if not isinstance(generated_ids, Tensor):
    #         raise TypeError('answer is not a Tensor')
            
    #     generated_ids_list: list[int] = generated_ids.tolist() # pyright: ignore[reportUnknownMemberType]

    #     self.debug_token_ids.extend(generated_ids_list)

    #     answer = self.qwen.tokenizer.decode( # pyright: ignore[reportUnknownMemberType]
    #         token_ids=generated_ids,
    #         skip_special_tokens=False,
    #     )
    #     if not isinstance(answer, str):
    #         raise TypeError('answer is not a str')

    #     self.messages.append({
    #         "role": RoleEnum.ASSISTANT,
    #         "content": answer,
    #     })

    #     print(self.messages)
    #     return answer

    def generate_internal_response(self, prompt: str):
        return self.generate_with_token_history(prompt=prompt, role=RoleEnum.SYSTEM)

    def show_tokens(self) -> None:
        print(self.debug_token_ids)        
    
    def append_chat_separator(self) -> None:
        tokens = self.qwen.tokenizer(
            text='\n',
            add_special_tokens=False,
        )
        separator_ids = tokens['input_ids']
        self.token_history.extend(separator_ids)
