from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        row = {key: _stringify(value) for key, value in row.items()}
        if self._fieldnames is None:
            if self.path.exists() and self.path.stat().st_size > 0:
                with self.path.open("r", newline="") as handle:
                    reader = csv.reader(handle)
                    self._fieldnames = next(reader)
            else:
                self._fieldnames = list(row.keys())

        fieldnames = self._fieldnames
        extra = [key for key in row if key not in fieldnames]
        if extra:
            fieldnames.extend(extra)

        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


class WandbLogger:
    def __init__(self, enabled: bool, project: str, run_name: str, config: dict[str, Any]):
        self.enabled = enabled
        self._wandb = None
        if enabled:
            import wandb

            self._wandb = wandb
            wandb.init(project=project, name=run_name, config=config)

    def log(self, row: dict[str, Any]) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.log(row)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()


def _stringify(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)

