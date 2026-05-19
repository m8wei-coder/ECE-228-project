from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED_COLUMNS = 26


def count_rows_and_check_columns(path: Path) -> tuple[int, int]:
    rows = 0
    first_width = -1
    with path.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            width = len(stripped.split())
            if first_width < 0:
                first_width = width
            if width != first_width:
                raise ValueError(f"{path} has inconsistent row width at data row {rows + 1}.")
            rows += 1
    return rows, first_width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="CMaps")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        train_path = data_dir / f"train_{subset}.txt"
        test_path = data_dir / f"test_{subset}.txt"
        rul_path = data_dir / f"RUL_{subset}.txt"
        for path, expected_width in [
            (train_path, EXPECTED_COLUMNS),
            (test_path, EXPECTED_COLUMNS),
            (rul_path, 1),
        ]:
            rows, width = count_rows_and_check_columns(path)
            if width != expected_width:
                raise ValueError(f"{path} has {width} columns, expected {expected_width}.")
            print(f"{path}: rows={rows}, columns={width}")


if __name__ == "__main__":
    main()

