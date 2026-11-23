from .config import MODEL_NAME, local_files_only, dtype, device_map, seeds_amount
from .qwen.qwen_factory import QwenFactory
from .experiment_starter import start_all_experiments


def main() -> None:
    qwen_factory = QwenFactory(
        model_name=MODEL_NAME,
        local_files_only=local_files_only,
        dtype=dtype,
        device_map=device_map,
    )
    qwen = qwen_factory.create_qwen()    
    start_all_experiments(qwen=qwen, seeds_amount=seeds_amount)


if __name__ == "__main__":
    main()
