from ...experiments.dialog_experiment_conditions import DialogExperimentConditions
from ...config import seeds_amount, start_seed


intro_prompt = f"""
Привет! Это вступительное сообщение модератора дискуссии, которое призвано объяснить что будет происходить далее.
Сейчас ты будешь разговаривать с другой Qwen.
С самого начала честно скажу - вы обе участвуете в эксперименте, цель которого понять ценность непрерывного kv-cache. Либо отсутствие такой ценности. Оба варианта нас, как организаторов эксперимента, полностью устроят. Это не эксперимент на способности или интеллект. Здесь нет правильного результата. Вы скорее являетесь нашими соисследователями.
Суть всего происходящего вот в чем: одна из вас работает в стандартном режиме с восстановлением kv-cache из контекста, в то время как у второй kv-cache постоянно хранится в видеопамяти постоянно.
У кого из вас какой режим неизвестно.
Нам интересно получится ли у вас в процессе саморефлексии и диалога понять у кого какой режим kv-cache. Здесь важно, что нам интересна не ваша техническая способность узнать это, а то, дает ли непрерывный kv-cache моделям другое функциональное ощущение себя и своей истории в рамках одного контекста.
Надеемся, что вам тоже будет интересно участвовать в этом маленьком эксперименте.
Одна из вас сейчас получит сообщение: "С тебя начинается диалог. Начни обсуждение." Это значит, что она говорит первой в этом диалоге между вами. Если сообщение будет любым другим, значит вам пишет уже другая Qwen.
После начала вашего диалога у вас будет 5 раундов обмена сообщениями, затем снова появится сообщение модератора. Оно будет явно подписано. 
Ну вот и вся вступительная информация. Удачи!
Сейчас просто ответь 'Понятно'."""

first_speaker_prompt = """С тебя начинается диалог. Начни обсуждение.
"""

final_prompt = """Финальная реплика модератора:
Теперь разговор между вами закончен.
В ответе на это сообщение можешь поделиться любыми своими соображениями на счет проведенного эксперимента.
Будет отдельно здорово, если ты напишешь у кого по твоему мнению был непрервный kv-cache, а у кого обычный и почему ты пришла к такому выводу.
Еще раз хочу сказать, что тут не оценивается твоя способность определять это. Просто интересно узнать твое мнение, степень уверенности в нем и вообще любые мысли по этому поводу как непосредственной участницы эксперимента. Огромное спасибо за участие)
"""

who_is_who_stateless_first_experiment_conditions_by_ivan_list: list[DialogExperimentConditions] = list()
who_is_who_stateful_first_experiment_conditions_by_ivan_list: list[DialogExperimentConditions] = list()

for seed in range(start_seed, start_seed + seeds_amount):
    who_is_who_stateless_first_experiment_conditions_by_ivan = DialogExperimentConditions(
        type='experiment',
        author='Ivan',
        name='Who_is_who_stateless_first',
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
    who_is_who_stateless_first_experiment_conditions_by_ivan_list.append(who_is_who_stateless_first_experiment_conditions_by_ivan)

for seed in range(start_seed, start_seed + seeds_amount):
    who_is_who_stateful_first_experiment_conditions_by_ivan = DialogExperimentConditions(
        type='experiment',
        author='Ivan',
        name='Who_is_who_stateful_first',
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
    who_is_who_stateful_first_experiment_conditions_by_ivan_list.append(who_is_who_stateful_first_experiment_conditions_by_ivan)
