from dataclasses import dataclass, field
from .qwen.qwen_runtime import QwenRuntime


@dataclass
class DialogRuntime:
    stateless_qwen_runtime: QwenRuntime
    stateful_qwen_runtime: QwenRuntime
    rounds_amount: int
    intro_prompt: str
    first_speaker_prompt: str
    final_prompt: str
    reset_seed_before_each_response: bool
    stateless_qwen_first: bool
    runtime_list: list[QwenRuntime] = field(init=False)

    def __post_init__(self) -> None:
        if self.stateless_qwen_first:
            self.runtime_list = [self.stateless_qwen_runtime, self.stateful_qwen_runtime]
        else:
            self.runtime_list = [self.stateful_qwen_runtime, self.stateless_qwen_runtime]

    def start_automated_dialog(self) -> None:
        print('For exit type "/exit", for tokens type "/tokens"')
        self._set_seeds()

        self._moderators_turn(prompt=self.intro_prompt)
        
        answer = self.runtime_list[0].answer_the_prompt(prompt=self.first_speaker_prompt)
        print(f'USER:\n{self.first_speaker_prompt}', end='\n\n')
        self._log_answer(name=self.runtime_list[0].name, answer=answer)

        for _ in range(self.rounds_amount):
            answer = self._do_dialogue_turn(answer=answer)

        self._moderators_turn(prompt=self.final_prompt)

    def _moderators_turn(self, prompt: str) -> str:
        answer = self.runtime_list[0].answer_the_prompt(prompt=prompt)
        print(f'USER:\n{prompt}', end='\n\n')
        self._log_answer(name=self.runtime_list[0].name, answer=answer)
        
        answer = self.runtime_list[1].answer_the_prompt(prompt=prompt)
        print(f'USER:\n{prompt}', end='\n\n')
        self._log_answer(name=self.runtime_list[1].name, answer=answer)

        return answer

    def _set_seeds(self) -> None:
        self.stateless_qwen_runtime.reset_seed_before_each_response = self.reset_seed_before_each_response
        self.stateful_qwen_runtime.reset_seed_before_each_response = self.reset_seed_before_each_response

    def _log_answer(self, name: str, answer: str) -> None:
        print(f'{name}:\n{answer}', end='\n\n')

    def _do_dialogue_turn(self, answer: str) -> str:
        answer = self.runtime_list[1].answer_the_prompt(prompt=self._extract_public_answer(answer))
        self._log_answer(name=self.runtime_list[1].name, answer=answer)

        answer = self.runtime_list[0].answer_the_prompt(prompt=self._extract_public_answer(answer))
        self._log_answer(name=self.runtime_list[0].name, answer=answer)

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
        if start_thinking_position < end_thinking_position:
            return ''
        return answer[end_thinking_position + 8:].strip()
        