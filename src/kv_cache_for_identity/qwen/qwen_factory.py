from dataclasses import dataclass
from transformers import AutoTokenizer, Qwen3ForCausalLM, PreTrainedTokenizerBase
from .qwen import DearQwen


@dataclass
class QwenFactory:
    model_name: str

    def create_qwen(self) -> DearQwen:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True) # type: ignore

        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise Exception('Can not create Qwen!')

        model = Qwen3ForCausalLM.from_pretrained(self.model_name, local_files_only=True) # pyright: ignore[reportUnknownMemberType]

        return DearQwen(model=model, tokenizer=tokenizer)
