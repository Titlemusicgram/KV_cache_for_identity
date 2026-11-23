import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime

from .log_event import LogEvent
from ..experiments.dialog_experiment_conditions import DialogExperimentConditions
from ..experiments.comparison_experiment_conditions import ComparisonExperimentConditions


@dataclass
class ExperimentLogger:
    conditions: DialogExperimentConditions | ComparisonExperimentConditions
    time_stamp: datetime = field(default_factory=datetime.now)
    file_path: Path = field(init=False)

    def __post_init__(self) -> None:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        self.file_path = logs_dir / f'{self.conditions.author}_{self.conditions.name}_seed_{self.conditions.seed}_{self.time_stamp.strftime("%Y-%m-%d_%H-%M-%S")}.jsonl'
        self._write_to_jsonl(event=self.conditions)
    
    def log(self, event: LogEvent) -> None:
        self._print_to_cli(event=event)
        self._write_to_jsonl(event=event)

    @staticmethod
    def _print_to_cli(event: LogEvent) -> None:
        print(f'{event.speaker}:\n{event.raw_output}', end='\n\n')

    def _write_to_jsonl(self, event: LogEvent | DialogExperimentConditions | ComparisonExperimentConditions) -> None:
        event_dict = asdict(event)
        json_string = json.dumps(event_dict, ensure_ascii=False)
        with open(self.file_path, "a", encoding='utf-8') as file:
            file.write(json_string + "\n")
