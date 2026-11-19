from .config import MODEL_NAME, local_files_only, dtype, device_map
from .qwen.qwen_factory import QwenFactory
from .experiments.experiments import start_comparison_experiment, start_dialog_experiment
from .experiments_list.Lira_Game_experiment import comparison_experiment_conditions_by_lira
from .experiments_list.Lira_Azimov_experiment import dialog_experiment_conditions_by_lira


def main() -> None:
    qwen_factory = QwenFactory(
        model_name=MODEL_NAME,
        local_files_only=local_files_only,
        dtype=dtype,
        device_map=device_map,
    )
    qwen = qwen_factory.create_qwen()

    start_comparison_experiment(qwen=qwen, conditions=comparison_experiment_conditions_by_lira)
    start_dialog_experiment(qwen=qwen, conditions=dialog_experiment_conditions_by_lira)


if __name__ == "__main__":
    main()
