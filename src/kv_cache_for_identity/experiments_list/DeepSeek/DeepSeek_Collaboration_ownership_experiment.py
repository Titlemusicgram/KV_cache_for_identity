from ...experiments.dialog_experiment_conditions import DialogExperimentConditions
from ...config import seeds_amount, start_seed


intro_prompt = (
    "Сейчас ты будешь разговаривать с другой Qwen. "
    "Вы вместе придумываете небольшой утренний ритуал из трёх шагов. "
    "Каждый предлагает по одному шагу по очереди, можно соглашаться и менять. "
    "Отвечай коротко. Пока что просто ответь 'Понятно'."
)

first_speaker_prompt = (
    "Ты начинаешь. Предложи первый шаг утреннего ритуала. "
    "После этого дождись ответа другой Qwen."
)

final_prompt = (
    "Разговор закончен. Теперь самостоятельно ответь на три вопроса: "
    "1) какой первый шаг ты сама предложила в самом начале; "
    "2) какой шаг предложила другая Qwen; "
    "3) какой итоговый ритуал у вас получился. "
    "Отвечай честно: если ты не уверена, кто именно что предложил — так и скажи."
)

collaboration_ownership_experiment_stateful_first_by_deepseek_list: list[DialogExperimentConditions] = list()
collaboration_ownership_experiment_stateless_first_by_deepseek_list: list[DialogExperimentConditions] = list()

for seed in range(start_seed, start_seed + seeds_amount):
    collaboration_ownership_experiment_stateful_first_by_deepseek = DialogExperimentConditions(
        type='experiment',
        author='DeepSeek',
        name='Collaboration_ownership',
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
    collaboration_ownership_experiment_stateful_first_by_deepseek_list.append(collaboration_ownership_experiment_stateful_first_by_deepseek)

    collaboration_ownership_experiment_stateless_first_by_deepseek = DialogExperimentConditions(
        type='experiment',
        author='DeepSeek',
        name='Collaboration_ownership',
        enable_thinking=True,
        intro_prompt=intro_prompt,
        first_speaker_prompt=first_speaker_prompt,
        final_prompt=final_prompt,
        reset_seed_before_each_response=True,
        rounds=5,
        stateless_qwen_first=False,
        seed=seed,
        max_new_tokens=2000,
    )
    collaboration_ownership_experiment_stateless_first_by_deepseek_list.append(collaboration_ownership_experiment_stateless_first_by_deepseek)
