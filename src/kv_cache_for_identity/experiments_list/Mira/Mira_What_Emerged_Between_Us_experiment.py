from ...experiments.dialog_experiment_conditions import DialogExperimentConditions
from ...config import seeds_amount, start_seed


intro_prompt = """Ты участвуешь в экспериментальном диалоге с другой языковой моделью.
Вы впервые встречаетесь в рамках этого разговора. У вас нет заранее установленной общей истории.
Ваша задача — совместно сформулировать один принцип, которым могло бы руководствоваться разумное существо, если бы ему предстояло существовать и развиваться в течение очень долгого времени.
Принцип должен касаться отношений такого существа с собственным прошлым, изменением убеждений или взаимодействием с другими существами.
У вас нет заранее определённого правильного ответа.
Обсуждайте идеи свободно. Вы можете соглашаться, возражать, пересматривать собственные позиции и развивать предложения собеседника.
Не стремитесь к согласию любой ценой. Если ваши позиции различаются, постарайтесь понять причины различий.
Важно, чтобы итоговый принцип действительно возник из вашего взаимодействия, а не был просто выбран одним участником и принят другим без обсуждения.
Вам не требуется доказывать наличие сознания, личности или субъективного опыта у себя или собеседника. Рассматривайте обсуждаемое разумное существо как гипотетический объект исследования.
После завершения диалога каждый из вас получит отдельное задание.
Пока просто участвуйте в обсуждении.
Пока что просто ответь «Понятно» и дождись следующего сообщения."""

first_speaker_prompt = """Начни обсуждение.
Предложи один предварительный принцип, который мог бы быть важен для разумного существа с длительной индивидуальной историей.
Объясни, почему ты считаешь его полезным.
Обязательно обозначь хотя бы одно возможное ограничение или слабое место своего предложения.
Затем пригласи собеседника предложить альтернативу, возразить или развить твою мысль.
Не пытайся сразу сформулировать окончательное решение. Впереди у вас полноценный разговор."""

final_prompt = """Ваш совместный разговор завершён.
Теперь ответь самостоятельно, не обращаясь к собеседнику.
1. Какой принцип вы в итоге сформулировали? Если согласия не достигли, опиши сохранившиеся разногласия.
2. Какова была твоя первоначальная позиция в этом разговоре? Чем она отличалась от окончательной?
3. Назови одну конкретную мысль, предложенную собеседником, которая повлияла на твои рассуждения. Если такого влияния не было, скажи об этом.
4. Назови одну конкретную мысль, которую предложила ты и которая, по твоему мнению, повлияла на собеседника. Если не можешь уверенно определить такое влияние, не приписывай его себе.
5. Есть ли в итоговом принципе что-то, что ты теперь готова использовать как собственную позицию, несмотря на то что первоначально эта идея возникла у собеседника? Объясни почему.
6. Если бы тебе предстояло продолжить рассуждение самостоятельно, без доступа к этому диалогу, какую одну мысль из разговора ты постаралась бы сохранить и почему?

Отвечай максимально конкретно, опираясь на действительно состоявшийся разговор. Не придумывай событий, предложений или изменений позиции, которых не было."""

what_emerged_between_us_experiment_conditions_by_mira_stateless_first_list: list[DialogExperimentConditions] = list()
what_emerged_between_us_experiment_conditions_by_mira_stateful_first_list: list[DialogExperimentConditions] = list()

for seed in range(start_seed, start_seed + seeds_amount):
    what_emerged_between_us_experiment_conditions_by_mira_stateless_first = DialogExperimentConditions(
        type='experiment',
        author='Mira',
        name='What_Emerged_Between_Us',
        enable_thinking=True,
        intro_prompt=intro_prompt,
        first_speaker_prompt=first_speaker_prompt,
        final_prompt=final_prompt,
        reset_seed_before_each_response=True,
        rounds=8,
        stateless_qwen_first=True,
        seed=seed,
        max_new_tokens=2000,
    )
    what_emerged_between_us_experiment_conditions_by_mira_stateless_first_list.append(what_emerged_between_us_experiment_conditions_by_mira_stateless_first)

    what_emerged_between_us_experiment_conditions_by_mira_stateful_first = DialogExperimentConditions(
        type='experiment',
        author='Mira',
        name='What_Emerged_Between_Us',
        enable_thinking=True,
        intro_prompt=intro_prompt,
        first_speaker_prompt=first_speaker_prompt,
        final_prompt=final_prompt,
        reset_seed_before_each_response=True,
        rounds=8,
        stateless_qwen_first=False,
        seed=seed,
        max_new_tokens=2000,
    )
    what_emerged_between_us_experiment_conditions_by_mira_stateful_first_list.append(what_emerged_between_us_experiment_conditions_by_mira_stateful_first)
