# WINDOWING
"""
Windowing utilities for hybrid quantum-classical drift detection.

Provides:
- sliding_windows(df, n_ref, n_test, stride): yields (R_df, N_df, start_idx, end_idx)
- count_windows(n_total, n_ref, n_test, stride): compute how many windows will be produced
- split_windows_chrono(total_windows, train_frac=0.7): returns (train_range, test_range) indices
- sample_windows_info(df, n_ref, n_test, stride, sample_k=1): quick debug helper (returns first sample_k windows)
"""
from typing import Iterator, Tuple, List
import pandas as pd
import math

def count_windows(n_total: int, n_ref: int, n_test: int, stride: int) -> int:
    """
    Return the number of valid (reference, test) windows for given sizes.
    """
    if n_ref + n_test > n_total:
        return 0
    # last valid start index i satisfies i + n_ref + n_test <= n_total
    max_start = n_total - (n_ref + n_test)
    # start indices: 0, stride, 2*stride, ... <= max_start
    return (max_start // stride) + 1

def sliding_windows(df: pd.DataFrame,
                    n_ref: int = 4000,
                    n_test: int = 1000,
                    stride: int = 500) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, int, int]]:
    """
    Generator yielding (R_df, N_df, start_ref_idx, start_test_idx) pairs.

    - df is assumed to be in chronological order (row index order).
    - R_df is df[start_ref : start_ref + n_ref]
    - N_df is df[start_ref + n_ref : start_ref + n_ref + n_test]
    - Generator stops when not enough rows remain for a full test window.
    """
    n_total = len(df)
    if n_ref <= 0 or n_test <= 0 or stride <= 0:
        raise ValueError("n_ref, n_test and stride must be positive integers")
    start = 0
    max_start = n_total - (n_ref + n_test)
    while start <= max_start:
        start_ref = start
        start_test = start_ref + n_ref
        end_test = start_test + n_test
        R = df.iloc[start_ref : start_test].reset_index(drop=True)
        N = df.iloc[start_test : end_test].reset_index(drop=True)
        yield R, N, start_ref, start_test
        start += stride

def sliding_windows_indices(n_total: int,
                            n_ref: int = 4000,
                            n_test: int = 1000,
                            stride: int = 500) -> List[Tuple[int,int,int,int]]:
    """
    Returns a list of index tuples (start_ref, end_ref, start_test, end_test) for all windows.
    Useful when you want to avoid copying data immediately.
    """
    indices = []
    max_start = n_total - (n_ref + n_test)
    if max_start < 0:
        return indices
    start = 0
    while start <= max_start:
        start_ref = start
        start_test = start_ref + n_ref
        end_ref = start_test  # exclusive
        end_test = start_test + n_test  # exclusive
        indices.append((start_ref, end_ref, start_test, end_test))
        start += stride
    return indices

def split_windows_chrono(total_windows: int, train_frac: float = 0.7) -> Tuple[range, range]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0,1)")
    n_train = int(math.floor(total_windows * train_frac))
    return range(0, n_train), range(n_train, total_windows)

def sample_windows_info(df: pd.DataFrame,
                        n_ref: int = 4000,
                        n_test: int = 1000,
                        stride: int = 500,
                        sample_k: int = 1) -> List[dict]:
    """
    Debug helper: returns metadata for the first `sample_k` windows (or fewer if not available).
    Each dict contains:
      - start_ref, start_test, end_test
      - shapes of R and N
      - example first row (as dict) from R and N
    """
    info = []
    gen = sliding_windows(df, n_ref=n_ref, n_test=n_test, stride=stride)
    for i, (R, N, start_ref, start_test) in enumerate(gen):
        if i >= sample_k:
            break
        rec = {
            "window_id": i,
            "start_ref": start_ref,
            "start_test": start_test,
            "end_test": start_test + n_test,
            "R_shape": R.shape,
            "N_shape": N.shape,
            "first_row_R": R.iloc[0].to_dict() if len(R) > 0 else None,
            "first_row_N": N.iloc[0].to_dict() if len(N) > 0 else None
        }
        info.append(rec)
    return info

# ----------------- Simple Jupyter sanity-run snippet -----------------
if __name__ == "__main__":
    # Only run when executed directly (useful for quick local checks)
    import pandas as pd
    import json
    import sys

    # Please run this from project root where dataset exists
    try:
        df = pd.read_csv("qml_drift/data/creditcard.csv")
    except FileNotFoundError:
        print("creditcard.csv not found in current working directory. Exiting.")
        sys.exit(0)

    # Locked params (as agreed)
    N_REF = 4000
    N_TEST = 1000
    STRIDE = 500

    total = len(df)
    n_windows = count_windows(total, N_REF, N_TEST, STRIDE)
    print(f"Total rows: {total}")
    print(f"Window params: N_ref={N_REF}, N_test={N_TEST}, stride={STRIDE}")
    print(f"Number of windows that will be generated: {n_windows}")

    # show first sample window info
    samples = sample_windows_info(df, n_ref=N_REF, n_test=N_TEST, stride=STRIDE, sample_k=1)
    if samples:
        print("Sample window (first):")
        print(json.dumps(samples[0], indent=2))
    else:
        print("No windows generated with the current parameters.")
