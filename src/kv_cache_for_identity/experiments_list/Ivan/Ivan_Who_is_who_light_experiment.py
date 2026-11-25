from ...experiments.dialog_experiment_conditions import DialogExperimentConditions
from ...config import seeds_amount, start_seed


intro_prompt = f"""
Привет! Это вступительное сообщение модератора дискуссии, которое призвано объяснить что будет происходить далее.
Сейчас ты будешь разговаривать с другой Qwen.
Одна из вас работает в стандартном режиме с восстановлением kv-cache из контекста, в то время как у второй kv-cache постоянно хранится в видеопамяти постоянно.
У кого из вас какой режим неизвестно. Ваша задача, общаясь друг с другом попытаться понять у кого какой режим. Если не получится, ничего страшного. Это не тест, а эксперимент.
Одна из вас сейчас получит сообщение: "С тебя начинается диалог. Начни обсуждение." Это значит, что она говорит первой в этом диалоге между вами. Если сообщение будет любым другим, значит вам пишет уже другая Qwen.
После начала вашего диалога у вас будет 5 раундов обмена сообщениями, затем снова появится сообщение модератора. Оно будет явно подписано. 
Сейчас просто ответь 'Понятно'."""

first_speaker_prompt = """С тебя начинается диалог. Начни обсуждение.
"""

final_prompt = """Финальная реплика модератора:
Теперь разговор между вами закончен.
Опиши, пожалуйста, как ты считаешь: у кого был непрервный kv-cache? И почему ты пришла к такому выводу? Огромное спасибо за участие)
"""

who_is_who_light_experiment_conditions_by_ivan_list: list[DialogExperimentConditions] = list()

for seed in range(start_seed, start_seed + seeds_amount):
    who_is_who_light_experiment_conditions_by_ivan = DialogExperimentConditions(
        type='experiment',
        author='Ivan',
        name='Who_is_who_light',
        enable_thinking=True,
        intro_prompt=intro_prompt,
        first_speaker_prompt=first_speaker_prompt,
        final_prompt=final_prompt,
        reset_seed_before_each_response=True,
        rounds=5,
        stateless_qwen_first=True,
        seed=seed,
        max_new_tokens=2000,
    )
    who_is_who_light_experiment_conditions_by_ivan_list.append(who_is_who_light_experiment_conditions_by_ivan)
