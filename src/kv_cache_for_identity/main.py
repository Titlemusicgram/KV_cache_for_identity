from .config import MODEL_NAME
from .qwen.qwen_factory import QwenFactory
from .experiments import start_comparison_experiment, start_dialog_experiment
from .dialog_experiment_conditions import DialogExperimentConditions
from .prompts import prompts, intro_prompt, first_speaker_prompt, final_prompt


def main() -> None:
    qwen_factory = QwenFactory(MODEL_NAME)
    qwen = qwen_factory.create_qwen()

    enable_thinking = True

    start_comparison_experiment(
        qwen=qwen,
        enable_thinking=enable_thinking,
        prompts=prompts,
    )

    dialog_experiment_conditions_by_lira = DialogExperimentConditions(
        author='Lira',
        name='Azimov_experiment',
        enable_thinking=enable_thinking,
        intro_prompt=intro_prompt,
        first_speaker_prompt=first_speaker_prompt,
        final_prompt=final_prompt,
        reset_seed_before_each_response=True,
        rounds=3,
        stateless_qwen_first=True,
    )

    start_dialog_experiment(
        qwen=qwen,
        conditions=dialog_experiment_conditions_by_lira,
    )


if __name__ == "__main__":
    main()
