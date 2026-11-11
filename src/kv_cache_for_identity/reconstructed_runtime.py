from .qwen.stateless_qwen_runtime import StatelessQwenRuntime
from .qwen.message_role_enum import RoleEnum


def reconstructed_runtime(
    qwen_runtime: StatelessQwenRuntime,
) -> None:
    print('For exit type "/exit", for tokens type "/tokens"')

    while True:
        prompt = input('USER> ')
        if prompt == '/exit':
            break
        if prompt == '/tokens':
            qwen_runtime.show_tokens()
            continue

        answer = qwen_runtime.generate_with_token_history(prompt=prompt, role=RoleEnum.USER)
        print(f'QWEN:\n{answer}', end='\n\n')
