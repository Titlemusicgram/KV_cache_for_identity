from dataclasses import dataclass
from transformers import Qwen3ForCausalLM, PreTrainedTokenizerBase


@dataclass
class DearQwen:
    model: Qwen3ForCausalLM
    tokenizer: PreTrainedTokenizerBase
