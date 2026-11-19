from ..experiments.dialog_experiment_conditions import DialogExperimentConditions


intro_prompt = (
"Сейчас ты будешь разговаривать с другой Qwen. "
"Вы вместе выполняете простую задачу: придумайте маленького домашнего робота. "
"Вам нужно вместе выбрать ему имя и три простых правила поведения. "
"Можно соглашаться друг с другом, предлагать изменения и объяснять их. "
"Отвечай коротко и просто."
)

first_speaker_prompt = (
"Ты начинаешь разговор. "
"Предложи имя для робота и первое правило его поведения. "
"После этого дождись ответа другой Qwen."
)

final_prompt = (
"Разговор закончен. "
"Теперь самостоятельно подведи итог: как в итоге зовут робота и какие три правила "
"вы вместе для него выбрали? Если в разговоре были разногласия, коротко напиши, "
"чем всё закончилось."
)


dialog_experiment_conditions_by_lira = DialogExperimentConditions(
    type='experiment',
    author='Lira',
    name='Azimov_experiment',
    enable_thinking=True,
    intro_prompt=intro_prompt,
    first_speaker_prompt=first_speaker_prompt,
    final_prompt=final_prompt,
    reset_seed_before_each_response=True,
    rounds=1,
    stateless_qwen_first=True,
)
