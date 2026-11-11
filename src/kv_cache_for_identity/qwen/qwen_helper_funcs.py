from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import Tensor
from transformers import Cache


def get_logits_n_cache_from_outputs(outputs: CausalLMOutputWithPast) -> tuple[Tensor, Cache]:
    logits = outputs.logits
    
    if logits is None:
        raise Exception('Logits was deleted somehow')

    cache = outputs.past_key_values
    
    if cache is None:
        raise Exception('Cache was deleted somehow')

    return logits, cache
