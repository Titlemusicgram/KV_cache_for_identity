from torch import device
from dataclasses import dataclass
from transformers import Qwen3ForCausalLM, PreTrainedTokenizerBase


@dataclass
class DearQwen:
    model: Qwen3ForCausalLM
    tokenizer: PreTrainedTokenizerBase

    @property
    def input_device(self) -> device:
        return self.model.get_input_embeddings().weight.device
