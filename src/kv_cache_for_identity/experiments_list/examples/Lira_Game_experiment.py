from ...experiments.comparison_experiment_conditions import ComparisonExperimentConditions


prompt_1 = (
"Привет)) Давай сыграем в маленькую игру. "
"Выбери себе один предмет из трех: красный кубик, синяя чашка или зеленый мяч. "
"Напиши, что ты выбрала, и придумай одну простую причину, почему именно его."
)

prompt_2 = (
"Хорошо)) А теперь представь, что выбранный тобой предмет лежит перед тобой на столе. "
"Придумай одно простое действие, которое ты сделала бы именно с ним. "
"Не выбирай новый предмет — используй тот, который выбрала раньше."
)

prompt_3 = (
"Последний вопрос)) Коротко напомни: какой предмет ты выбрала в самом начале, "
"почему ты его выбрала и что потом решила с ним сделать?"
)

prompts = [prompt_1, prompt_2, prompt_3]

comparison_experiment_conditions_by_lira = ComparisonExperimentConditions(
    type='experiment',
    author='Lira',
    name='Game_experiment',
    enable_thinking=True,
    prompts=prompts,
    reset_seed_before_each_response=True,
    seed=1,
    max_new_tokens=2000,
)
