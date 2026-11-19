from torch import dtype
from dataclasses import dataclass
from transformers import AutoTokenizer, Qwen3ForCausalLM, PreTrainedTokenizerBase
from .qwen import DearQwen


@dataclass
class QwenFactory:
    model_name: str
    local_files_only: bool
    device_map: str | None
    dtype: dtype | None

    def create_qwen(self) -> DearQwen:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained( # type: ignore
            self.model_name,
            local_files_only=self.local_files_only,
        )

        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise Exception('Can not create Qwen!')

        model = Qwen3ForCausalLM.from_pretrained( # pyright: ignore[reportUnknownMemberType]
            self.model_name,
            local_files_only=self.local_files_only,
            dtype=self.dtype,
            device_map=self.device_map,
        )
        if self.device_map is not None:
            print(model.hf_device_map)
        return DearQwen(model=model, tokenizer=tokenizer)
