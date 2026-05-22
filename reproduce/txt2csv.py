"""
Convert raw C-MAPSS .txt files to CSV files that reproduce/*.py expects.

Source : ../CMaps/{train,test,RUL}_FD00X.txt   (repo root, committed by teammate)
Target : ./csv/{train,test,rul}/{train,test,RUL}_FD00X.csv

Column spec (confirmed against reproduce/data_preprocessing.py):
  train/test : 26 cols, header required
      unit_number, time_cycles,
      setting_1, setting_2, setting_3,
      sensor_1 ... sensor_21
  RUL        : 1 col, header 'true_rul' required
"""

import os
import sys
import pandas as pd

SRC_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "CMaps")
)
DST_ROOT = os.path.join(os.path.dirname(__file__), "csv")

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

COLUMNS = (
    ["unit_number", "time_cycles", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
RUL_COLUMN = "true_rul"


def count_txt_lines(path):
    with open(path, "r") as f:
        return sum(1 for line in f if line.strip() != "")


def read_trailing_clean(path):
    """
    Space-separated, trailing spaces produce all-NaN tail columns.
    Read with whitespace delimiter and drop any column that is fully NaN.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    return df


def convert_train_or_test(src_path, dst_path):
    df = read_trailing_clean(src_path)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{src_path}: expected {len(COLUMNS)} columns after trim, "
            f"got {df.shape[1]}"
        )
    df.columns = COLUMNS
    df["unit_number"] = df["unit_number"].astype(int)
    df["time_cycles"] = df["time_cycles"].astype(int)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    df.to_csv(dst_path, index=False)
    return df


def convert_rul(src_path, dst_path):
    df = read_trailing_clean(src_path)
    if df.shape[1] != 1:
        raise ValueError(
            f"{src_path}: expected 1 column after trim, got {df.shape[1]}"
        )
    df.columns = [RUL_COLUMN]
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    df.to_csv(dst_path, index=False)
    return df


def validate(subset, train_df, test_df, rul_df, train_txt, test_txt, rul_txt):
    rows = []

    for name, df, txt_path in [
        ("train", train_df, train_txt),
        ("test", test_df, test_txt),
    ]:
        assert list(df.columns) == COLUMNS, (
            f"{subset} {name}: column mismatch\n"
            f"  expected: {COLUMNS}\n  got:      {list(df.columns)}"
        )
        assert df.shape[1] == 26, f"{subset} {name}: ncols != 26"
        assert not df.isna().all(axis=0).any(), (
            f"{subset} {name}: found all-NaN column"
        )
        txt_lines = count_txt_lines(txt_path)
        assert len(df) == txt_lines, (
            f"{subset} {name}: row count {len(df)} != txt lines {txt_lines}"
        )
        non_numeric = [c for c in df.columns if df[c].dtype == object]
        assert not non_numeric, (
            f"{subset} {name}: non-numeric cols: {non_numeric}"
        )
        rows.append((subset, name, len(df), df.shape[1], "OK"))

    assert list(rul_df.columns) == [RUL_COLUMN], (
        f"{subset} rul: header != ['{RUL_COLUMN}']"
    )
    assert not rul_df.isna().all(axis=0).any(), (
        f"{subset} rul: all-NaN column"
    )
    assert pd.api.types.is_numeric_dtype(rul_df[RUL_COLUMN]), (
        f"{subset} rul: not numeric"
    )
    rul_txt_lines = count_txt_lines(rul_txt)
    assert len(rul_df) == rul_txt_lines, (
        f"{subset} rul: row count {len(rul_df)} != txt lines {rul_txt_lines}"
    )

    n_engines = test_df["unit_number"].nunique()
    if len(rul_df) != n_engines:
        raise AssertionError(
            f"{subset}: RUL rows ({len(rul_df)}) != "
            f"unique unit_number in test ({n_engines}). "
            f"Stopping. No data was auto-modified."
        )
    rows.append((subset, "rul", len(rul_df), 1, f"n_engines={n_engines} OK"))

    return rows


def main():
    if not os.path.isdir(SRC_DIR):
        print(f"ERROR: source dir not found: {SRC_DIR}", file=sys.stderr)
        sys.exit(1)

    all_rows = []

    for subset in SUBSETS:
        train_txt = os.path.join(SRC_DIR, f"train_{subset}.txt")
        test_txt = os.path.join(SRC_DIR, f"test_{subset}.txt")
        rul_txt = os.path.join(SRC_DIR, f"RUL_{subset}.txt")

        train_csv = os.path.join(DST_ROOT, "train", f"train_{subset}.csv")
        test_csv = os.path.join(DST_ROOT, "test", f"test_{subset}.csv")
        rul_csv = os.path.join(DST_ROOT, "rul", f"RUL_{subset}.csv")

        train_df = convert_train_or_test(train_txt, train_csv)
        test_df = convert_train_or_test(test_txt, test_csv)
        rul_df = convert_rul(rul_txt, rul_csv)

        rows = validate(
            subset, train_df, test_df, rul_df,
            train_txt, test_txt, rul_txt,
        )
        all_rows.extend(rows)

    print(f"\n{'subset':<8}{'kind':<6}{'rows':>8}{'cols':>6}  note")
    print("-" * 50)
    for subset, kind, n_rows, n_cols, note in all_rows:
        print(f"{subset:<8}{kind:<6}{n_rows:>8}{n_cols:>6}  {note}")
    print("\nAll subsets converted and validated.")


if __name__ == "__main__":
    main()
