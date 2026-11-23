from .dialog_experiment_conditions import DialogExperimentConditions
from .comparison_experiment_conditions import ComparisonExperimentConditions

from ..qwen.qwen import DearQwen
from ..qwen.stateful_qwen_runtime import StatefulQwenRuntime
from ..qwen.stateless_qwen_runtime import StatelessQwenRuntime
from ..runtimes.comparison_runtime import ComparisonRuntime
from ..runtimes.dialog_runtime import DialogRuntime
from ..logger.logger import ExperimentLogger


def start_comparison_experiment(
    qwen: DearQwen,
    conditions: ComparisonExperimentConditions,
) -> None:
    stateful_qwen_runtime = StatefulQwenRuntime(
        qwen=qwen,
        enable_thinking=conditions.enable_thinking,
        reset_seed_before_each_response=conditions.reset_seed_before_each_response,
        seed=conditions.seed,
        max_new_tokens=conditions.max_new_tokens,
    )

    stateless_qwen_runtime = StatelessQwenRuntime(
        qwen=qwen,
        enable_thinking=conditions.enable_thinking,
        reset_seed_before_each_response=conditions.reset_seed_before_each_response,
        seed=conditions.seed,
        max_new_tokens=conditions.max_new_tokens,
    )

    logger = ExperimentLogger(conditions=conditions)

    stateful_runtime = ComparisonRuntime(
        qwen_runtime=stateful_qwen_runtime,
        logger=logger,
        prompts=conditions.prompts,
    )

    stateless_runtime = ComparisonRuntime(
        qwen_runtime=stateless_qwen_runtime,
        logger=logger,
        prompts=conditions.prompts,
    )

    stateful_runtime.start_automated_dialog()
    stateless_runtime.start_automated_dialog()

def start_dialog_experiment(
    qwen: DearQwen,
    conditions: DialogExperimentConditions,
) -> None:
    stateful_qwen_runtime = StatefulQwenRuntime(
        qwen=qwen,
        enable_thinking=conditions.enable_thinking,
        reset_seed_before_each_response=conditions.reset_seed_before_each_response,
        seed=conditions.seed,
        max_new_tokens=conditions.max_new_tokens,
    )
    
    stateless_qwen_runtime = StatelessQwenRuntime(
        qwen=qwen,
        enable_thinking=conditions.enable_thinking,
        reset_seed_before_each_response=conditions.reset_seed_before_each_response,
        seed=conditions.seed,
        max_new_tokens=conditions.max_new_tokens,
    )

    logger = ExperimentLogger(conditions=conditions)

    dialog_runtime = DialogRuntime(
        stateless_qwen_runtime=stateless_qwen_runtime,
        stateful_qwen_runtime=stateful_qwen_runtime,
        rounds_amount=conditions.rounds,
        intro_prompt=conditions.intro_prompt,
        first_speaker_prompt=conditions.first_speaker_prompt,
        final_prompt=conditions.final_prompt,
        stateless_qwen_first=conditions.stateless_qwen_first,
        logger=logger,
    )

    dialog_runtime.start_automated_dialog()
