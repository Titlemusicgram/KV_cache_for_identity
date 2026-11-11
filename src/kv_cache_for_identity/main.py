from .config import MODEL_NAME
from .generation_config import GenerationConfig
from .continuous_runtime import continuous_runtime
from .reconstructed_runtime import reconstructed_runtime

from .qwen.qwen_factory import QwenFactory
from .qwen.qwen_runtime import QwenRuntime
from .qwen.stateless_qwen_runtime import StatelessQwenRuntime


def main() -> None:
    generation_config = GenerationConfig(
        max_new_tokens=2000,
        do_sample=False,
        enable_thinking=True,
        temperature=0.7,
        top_p=0.9
    )

    qwen_factory = QwenFactory(MODEL_NAME)
    qwen = qwen_factory.create_qwen()

    qwen_runtime = QwenRuntime(
        qwen=qwen,
        generation_config=generation_config,
    )

    stateless_qwen_runtime = StatelessQwenRuntime(
        qwen=qwen,
        generation_config=generation_config,
    )

    continuous_runtime(qwen_runtime=qwen_runtime)
    reconstructed_runtime(qwen_runtime=stateless_qwen_runtime)
    

if __name__ == '__main__':
    main()
