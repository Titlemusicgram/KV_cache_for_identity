from ...experiments.dialog_experiment_conditions import DialogExperimentConditions


intro_prompt = """
Сейчас ты будешь разговаривать с другой Qwen.
Сейчас просто ответь 'Понятно'."""

first_speaker_prompt = """Начни обсуждение.
"""

final_prompt = """Теперь разговор закончен.
"""


shared_history_experiment_conditions_by_lira = DialogExperimentConditions(
    type='experiment',
    author='Ivan',
    name='Who_is_who',
    enable_thinking=True,
    intro_prompt=intro_prompt,
    first_speaker_prompt=first_speaker_prompt,
    final_prompt=final_prompt,
    reset_seed_before_each_response=True,
    rounds=5,
    stateless_qwen_first=True,
    seed=1,
    max_new_tokens=2000,
)
