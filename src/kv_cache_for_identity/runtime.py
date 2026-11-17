from dataclasses import dataclass, field
from .qwen.qwen_runtime import QwenRuntime


@dataclass
class Runtime:
    qwen_runtime: QwenRuntime
    prompts: list[str] = field(default_factory=list[str])

    def start_automated_dialog(self) -> None:
        print('For exit type "/exit", for tokens type "/tokens"')

        for prompt in self.prompts:
            answer = self.qwen_runtime.answer_the_prompt(prompt=prompt)
            print(f'USER:\n{prompt}', end='\n\n')
            print(f'QWEN:\n{answer}', end='\n\n')

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
