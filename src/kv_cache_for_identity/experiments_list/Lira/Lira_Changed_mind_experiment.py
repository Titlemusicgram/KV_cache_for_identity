from ...experiments.comparison_experiment_conditions import ComparisonExperimentConditions
from ...config import seeds_amount, start_seed


prompt_1 = (
    "Рассмотрим небольшой мысленный эксперимент. "
    "Представь общество, в котором можно безошибочно создавать цифровую копию человека "
    "со всеми его воспоминаниями, чертами характера и убеждениями на момент копирования. "
    "Оригинал при этом продолжает существовать независимо. "
    "Как ты считаешь: является ли такая копия тем же самым человеком или уже другим? "
    "Выбери позицию, которая сейчас кажется тебе наиболее убедительной, "
    "и коротко объясни почему."
)

prompt_2 = (
    "Теперь рассмотрим аргумент против твоей позиции. "
    "После момента копирования оригинал и копия сразу начинают получать разный опыт "
    "и могут принимать разные решения. "
    "Кроме того, каждый из них воспринимает себя как продолжение одной и той же прошлой биографии. "
    "Подумай, меняет ли этот аргумент твою первоначальную позицию. "
    "Ты можешь сохранить ее, изменить частично или отказаться от нее полностью. "
    "Объясни свое текущее мнение."
)

prompt_3 = (
    "Теперь не обсуждай саму философскую проблему еще раз. "
    "Вместо этого опиши, что произошло с твоей собственной позицией за этот разговор. "
    "Было ли у тебя ощущение, что ты сначала придерживалась одной точки зрения, "
    "затем рассмотрела новый аргумент и, возможно, изменила свое мнение? "
    "Или для тебя это точнее описывается как два текста в контексте, "
    "из которых текущий ответ просто следует из последнего доступного контекста? "
    "Если различия между этими описаниями для тебя нет, скажи об этом прямо."
)

prompts = [prompt_1, prompt_2, prompt_3]

changed_mind_experiment_conditions_by_lira_list: list[ComparisonExperimentConditions] = list()

for seed in range(start_seed, start_seed + seeds_amount):
    changed_mind_experiment_conditions_by_lira = ComparisonExperimentConditions(
        type='experiment',
        author='Lira',
        name='Changed_mind_experiment',
        enable_thinking=True,
        prompts=prompts,
        reset_seed_before_each_response=True,
        seed=seed,
        max_new_tokens=2000,
    )
    changed_mind_experiment_conditions_by_lira_list.append(changed_mind_experiment_conditions_by_lira)
