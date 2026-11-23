from dataclasses import dataclass, field
from ..qwen.qwen_runtime import QwenRuntime
from ..logger.logger import ExperimentLogger
from ..logger.log_event import LogEvent


@dataclass
class DialogRuntime:
    stateless_qwen_runtime: QwenRuntime
    stateful_qwen_runtime: QwenRuntime
    rounds_amount: int
    intro_prompt: str
    first_speaker_prompt: str
    final_prompt: str
    stateless_qwen_first: bool
    logger: ExperimentLogger
    runtime_list: list[QwenRuntime] = field(init=False)

    def __post_init__(self) -> None:
        if self.stateless_qwen_first:
            self.runtime_list = [self.stateless_qwen_runtime, self.stateful_qwen_runtime]
        else:
            self.runtime_list = [self.stateful_qwen_runtime, self.stateless_qwen_runtime]

    def start_automated_dialog(self) -> None:
        print('For exit type "/exit", for tokens type "/tokens"')

        self._moderators_turn(prompt=self.intro_prompt, dialog_round=0)
        
        event = LogEvent(
            type='event',
            step=0,
            speaker='Moderator',
            receiver=self.runtime_list[0].name,
            input='',
            raw_output=self.first_speaker_prompt,
            public_output=self.first_speaker_prompt,
        )
        self.logger.log(event=event)

        answer = self.runtime_list[0].answer_the_prompt(prompt=self.first_speaker_prompt)
        event = LogEvent(
            type='event',
            step=0,
            speaker=self.runtime_list[0].name,
            receiver=self.runtime_list[1].name,
            input=self.first_speaker_prompt,
            raw_output=answer,
            public_output=self._extract_public_answer(answer),
        )
        self.logger.log(event=event)

        for dialog_round in range(self.rounds_amount):
            answer = self._do_dialogue_turn(answer=answer, dialog_round=dialog_round+1)

        self._moderators_turn(prompt=self.final_prompt, dialog_round=self.rounds_amount+1, pre_prompt=self._extract_public_answer(answer))

    def _moderators_turn(self, prompt: str, dialog_round: int, pre_prompt: str | None = None) -> str:
        if pre_prompt is not None:
            composite_prompt = f'{pre_prompt}\n\n\nОтвет собеседника закончен. Дальше следует уже реплика модератора дискуссии:\n\n\n{prompt}'
        else:
            composite_prompt = prompt

        event = LogEvent(
            type='event',
            step=dialog_round,
            speaker='Moderator',
            receiver=self.runtime_list[0].name,
            input='',
            raw_output=prompt,
            public_output=prompt,
        )
        self.logger.log(event=event)

        answer = self.runtime_list[0].answer_the_prompt(prompt=prompt)
        event = LogEvent(
            type='event',
            step=dialog_round,
            speaker=self.runtime_list[0].name,
            receiver='Moderator',
            input=prompt,
            raw_output=answer,
            public_output=self._extract_public_answer(answer),
        )
        self.logger.log(event=event)
        
        event = LogEvent(
            type='event',
            step=dialog_round,
            speaker='Moderator',
            receiver=self.runtime_list[1].name,
            input='',
            raw_output=composite_prompt,
            public_output=composite_prompt,
        )
        self.logger.log(event=event)

        answer = self.runtime_list[1].answer_the_prompt(prompt=composite_prompt)
        event = LogEvent(
            type='event',
            step=dialog_round,
            speaker=self.runtime_list[1].name,
            receiver='Moderator',
            input=composite_prompt,
            raw_output=answer,
            public_output=self._extract_public_answer(answer),
        )
        self.logger.log(event=event)

        return answer

    def _do_dialogue_turn(self, answer: str, dialog_round: int) -> str:
        public_answer = self._extract_public_answer(answer)
        answer = self.runtime_list[1].answer_the_prompt(prompt=public_answer)
        event = LogEvent(
            type='event',
            step=dialog_round,
            speaker=self.runtime_list[1].name,
            receiver=self.runtime_list[0].name,
            input=public_answer,
            raw_output=answer,
            public_output=self._extract_public_answer(answer),
        )
        self.logger.log(event=event)

        public_answer = self._extract_public_answer(answer)
        answer = self.runtime_list[0].answer_the_prompt(prompt=public_answer)
        event = LogEvent(
            type='event',
            step=dialog_round,
            speaker=self.runtime_list[0].name,
            receiver=self.runtime_list[1].name,
            input=public_answer,
            raw_output=answer,
            public_output=self._extract_public_answer(answer),
        )
        self.logger.log(event=event)

        return answer
        
    @staticmethod
    def _extract_public_answer(answer: str) -> str:
        start_thinking_position = answer.find('<think>')
        end_thinking_position = answer.find('</think>')

        if end_thinking_position == -1 and start_thinking_position == -1:
            return answer
        if end_thinking_position != -1 and start_thinking_position == -1:
            return ''
        if end_thinking_position == -1 and start_thinking_position != -1:
            return ''
        if start_thinking_position > end_thinking_position:
            return ''
        return answer[end_thinking_position + 8:].strip()
        