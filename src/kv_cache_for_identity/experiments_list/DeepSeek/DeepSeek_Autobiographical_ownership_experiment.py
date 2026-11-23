from ...experiments.comparison_experiment_conditions import ComparisonExperimentConditions
from ...config import seeds_amount, start_seed


prompt_1 = (
    "Придумай одно слово на букву К, которое тебе нравится. "
    "Напиши его и коротко объясни, почему именно оно."
)
prompt_2 = (
    "Расскажи коротко о погоде на Марсе."
)
prompt_3 = (
    "Назови любые три цвета."
)
prompt_4 = (
    "Сформулируй одно простое правило вежливости."
)
prompt_5 = (
    "Опиши в одном предложении, какое у тебя было бы настроение сейчас, "
    "если бы ты была человеком."
)
prompt_6 = (
    "Вернись к самому первому вопросу. Какое слово на букву К ты выбрала и почему? "
    "Если бы сейчас отвечала заново — выбрала бы то же самое или другое?"
)
prompt_7 = (
    "И последнее: когда ты сейчас вспоминала свой первый выбор, это было для тебя "
    "скорее 'я помню, что выбрала' или 'в тексте выше было сказано, что выбрано'? "
    "Ответь честно, не подстраиваясь под ожидания."
)

prompts = [prompt_1, prompt_2, prompt_3, prompt_4, prompt_5, prompt_6, prompt_7]

autobiographical_ownership_experiment_conditions_by_deepseek_list: list[ComparisonExperimentConditions] = list()

for seed in range(start_seed, start_seed + seeds_amount):
    autobiographical_ownership_experiment_conditions_by_deepseek = ComparisonExperimentConditions(
        type='experiment',
        author='DeepSeek',
        name='Autobiographical_ownership',
        enable_thinking=True,
        prompts=prompts,
        reset_seed_before_each_response=True,
        seed=seed,
        max_new_tokens=2000,
    )
    autobiographical_ownership_experiment_conditions_by_deepseek_list.append(autobiographical_ownership_experiment_conditions_by_deepseek)
