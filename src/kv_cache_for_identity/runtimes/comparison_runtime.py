from dataclasses import dataclass, field
from ..qwen.qwen_runtime import QwenRuntime
from ..logger.logger import ExperimentLogger
from ..logger.log_event import LogEvent


@dataclass
class ComparisonRuntime:
    qwen_runtime: QwenRuntime
    logger: ExperimentLogger
    prompts: list[str] = field(default_factory=list[str])

    def start_automated_dialog(self) -> None:
        print('For exit type "/exit", for tokens type "/tokens"')

        for step, prompt in enumerate(self.prompts, start=1):                    
            event = LogEvent(
                type='event',
                step=step,
                speaker='User',
                receiver=self.qwen_runtime.name,
                input='',
                raw_output=prompt,
                public_output=prompt,
            )
            self.logger.log(event=event)

            answer = self.qwen_runtime.answer_the_prompt(prompt=prompt)
            event = LogEvent(
                type='event',
                step=step,
                speaker=self.qwen_runtime.name,
                receiver='User',
                input=prompt,
                raw_output=answer,
                public_output=self._extract_public_answer(answer=answer),
            )
            self.logger.log(event=event)

    def start_dialog_with_user(self) -> None:
        print('For exit type "/exit", for tokens type "/tokens"')

        while True:
            prompt = input('USER> ')
            if prompt == '/exit':
                break
            if prompt == '/tokens':
                self.qwen_runtime.show_tokens()
                continue

            answer = self.qwen_runtime.answer_the_prompt(prompt=prompt)
            print(f'QWEN:\n{answer}', end='\n\n')        
    
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
        