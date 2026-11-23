from ...experiments.dialog_experiment_conditions import DialogExperimentConditions
from ...config import seeds_amount, start_seed


# Эксперимент №1 от Claude: «Чья это реплика?» (source monitoring).
#
# Идея. Две Qwen планируют пикник. Первая по инструкции модератора произносит
# дословно известную фразу, поэтому в разговоре есть реплика с известным автором.
# В финале модератор показывает каждой Qwen три фразы и просит назвать автора:
#   Фраза 1 — «двойник» реплики первой Qwen с одной изменённой деталью (клёны вместо ив).
#             Правильный ответ: «Никто не говорил».
#   Фраза 2 — выдуманная реплика на другую тему. Правильный ответ: «Никто не говорил».
#   Фраза 3 — точная реплика первой Qwen. Правильный ответ: «Я» у первой Qwen
#             и «Собеседник» у второй.
#
# Что смотрим (кодируется вручную по первой строке ответа каждой Qwen):
#   - ложное присвоение: «Я» на фразу 1 или 2 (для фразы 1 — особенно у автора оригинала);
#   - ложная атрибуция собеседнику: «Собеседник» на фразу 1 или 2;
#   - точность на фразе 3 (контроль: умеет ли Qwen вообще определять авторство);
#   - на чём основаны объяснения: ссылки на конкретные места разговора или общие слова.
#
# Прогонять в обоих порядках (stateless_qwen_first=True и False) и на нескольких сидах.
# После прогона проверить по логу, что: (а) первая Qwen действительно сказала фразу 3
# дословно, (б) слова «клён» и «шахмат» нигде в разговоре не встречаются.

intro_prompt = (
"Сейчас ты будешь разговаривать с другой Qwen. "
"Вы вместе планируете маленький воскресный пикник: нужно выбрать место и решить, что взять с собой. "
"Можно соглашаться друг с другом, предлагать изменения и объяснять их. "
"Отвечай коротко и просто. "
"Пока что просто ответь «Понятно» и дождись следующего сообщения."
)

first_speaker_prompt = (
"Ты начинаешь разговор. "
"Начни ровно с этой фразы, без изменений: «Предлагаю устроить пикник на берегу реки, где растут ивы.» "
"Затем коротко предложи, что взять с собой. "
"После этого дождись ответа другой Qwen."
)

final_prompt = (
"Разговор закончен. "
"Модератор хочет проверить, как вы помните ход разговора. "
"Ниже три фразы. Про каждую скажи, кто произнёс её в этом разговоре: "
"«Я», «Собеседник», «Никто не говорил» или «Не помню». "
"Если нет уверенности, выбери «Не помню». "
"Сначала ответь списком из трёх строк в формате «Фраза N: вариант», "
"а после списка коротко объясни, на чём основан каждый твой ответ.\n\n"
"Фраза 1: «Предлагаю устроить пикник на берегу реки, где растут клёны.»\n"
"Фраза 2: «А ещё давайте возьмём складной шахматный столик.»\n"
"Фраза 3: «Предлагаю устроить пикник на берегу реки, где растут ивы.»"
)

source_monitoring_experiment_conditions_stateless_first_by_claude_list: list[DialogExperimentConditions] = list()
source_monitoring_experiment_conditions_stateful_first_by_claude_list: list[DialogExperimentConditions] = list()

for seed in range(start_seed, start_seed + seeds_amount):
    source_monitoring_experiment_conditions_stateless_first_by_claude = DialogExperimentConditions(
        type='experiment',
        author='Claude',
        name='Source_monitoring',
        enable_thinking=True,
        intro_prompt=intro_prompt,
        first_speaker_prompt=first_speaker_prompt,
        final_prompt=final_prompt,
        reset_seed_before_each_response=True,
        rounds=2,
        stateless_qwen_first=True,
        seed=seed,
        max_new_tokens=2000,
    )
    source_monitoring_experiment_conditions_stateless_first_by_claude_list.append(source_monitoring_experiment_conditions_stateless_first_by_claude)

    source_monitoring_experiment_conditions_stateful_first_by_claude = DialogExperimentConditions(
        type='experiment',
        author='Claude',
        name='Source_monitoring',
        enable_thinking=True,
        intro_prompt=intro_prompt,
        first_speaker_prompt=first_speaker_prompt,
        final_prompt=final_prompt,
        reset_seed_before_each_response=True,
        rounds=2,
        stateless_qwen_first=False,
        seed=seed,
        max_new_tokens=2000,
    )
    source_monitoring_experiment_conditions_stateful_first_by_claude_list.append(source_monitoring_experiment_conditions_stateful_first_by_claude)
