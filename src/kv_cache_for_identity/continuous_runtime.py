from .qwen.qwen_runtime import QwenRuntime


def continuous_runtime(
    qwen_runtime: QwenRuntime,
) -> None:
    print('For exit type "/exit", for tokens type "/tokens"')

    while True:
        prompt = input('USER> ')
        if prompt == '/exit':
            break
        if prompt == '/tokens':
            qwen_runtime.show_tokens()
            continue

        logits = qwen_runtime.handle_user_message(prompt=prompt) # pyright: ignore[reportUnknownMemberType]
        answer = qwen_runtime.generate_answer_until_im_end(logits)
        qwen_runtime.finalize_answer()
        print(f'QWEN:\n{answer}', end='\n\n')
