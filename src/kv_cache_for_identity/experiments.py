from .qwen.qwen import DearQwen
from .qwen.stateful_qwen_runtime import StatefulQwenRuntime
from .qwen.stateless_qwen_runtime import StatelessQwenRuntime

from .runtime import Runtime
from .dialog_runtime import DialogRuntime
from .dialog_experiment_conditions import DialogExperimentConditions


def start_comparison_experiment(
    qwen: DearQwen,
    enable_thinking: bool,
    prompts: list[str],
) -> None:
    stateful_qwen_runtime = StatefulQwenRuntime(qwen=qwen, enable_thinking=enable_thinking)
    stateless_qwen_runtime = StatelessQwenRuntime(qwen=qwen, enable_thinking=enable_thinking)

    stateful_runtime = Runtime(qwen_runtime=stateful_qwen_runtime, prompts=prompts)
    stateless_runtime = Runtime(qwen_runtime=stateless_qwen_runtime, prompts=prompts)

    stateful_runtime.start_automated_dialog()
    stateless_runtime.start_automated_dialog()

def start_dialog_experiment(
    qwen: DearQwen,
    conditions: DialogExperimentConditions,
) -> None:
    stateful_qwen_runtime = StatefulQwenRuntime(qwen=qwen, enable_thinking=conditions.enable_thinking)
    stateless_qwen_runtime = StatelessQwenRuntime(qwen=qwen, enable_thinking=conditions.enable_thinking)

    dialog_runtime = DialogRuntime(
        stateless_qwen_runtime=stateless_qwen_runtime,
        stateful_qwen_runtime=stateful_qwen_runtime,
        rounds_amount=conditions.rounds,
        intro_prompt=conditions.intro_prompt,
        first_speaker_prompt=conditions.first_speaker_prompt,
        final_prompt=conditions.final_prompt,
        stateless_qwen_first=conditions.stateless_qwen_first,
        reset_seed_before_each_response=conditions.reset_seed_before_each_response,
    )

    dialog_runtime.start_automated_dialog()
