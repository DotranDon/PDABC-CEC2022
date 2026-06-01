# -*- coding: utf-8 -*-
"""
cec2022_target_reaching_score.py

Compute CEC2022-style target-reaching scores.

For each dimension and benchmark function, the script compares
all pairs of algorithms run by run. A trial that reaches EPS = 1e-8
is considered better than a trial that does not reach EPS. If both
trials reach EPS, the smaller FEterm is better. If neither trial
reaches EPS, the smaller final error is better. Exact ties give
0.5 point to each algorithm.

Input:
    AlgorithmName_FunctionNo_D.txt

Each input file is a 17 x 30 matrix:
    rows 0..15 : error values at CEC recording/checkpoint points
    row  16    : FEterm or checkpoint index in some original files

Output:
    CEC2022_Overall_Summary_Report.csv
    CEC2022_Function_Scores.csv
    CEC2022_Pairwise_Scores.csv
    CEC2022_All_Trial_Data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

EPS = 1e-8
N_RUNS = 30

MAXFES_BY_D = {
    10: 200_000,
    20: 1_000_000,
}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CEC2022 FILE LOADING
# ============================================================

def cec2022_record_points(D):
    """
    Return the 16 CEC2022 recording points used to convert checkpoint
    indices in original result files into function evaluation counts.

    Some official .txt result files do not store the actual FEterm value
    in row 17. Instead, they store a checkpoint index from 1 to 16, or 17.
    Following the logic in load_data.m, a checkpoint index is converted
    to the corresponding number of function evaluations as

    ```
    FE = ceil(10^((idx - 1)/5 - 3) * MaxFES)
    ```

    For k = 0,...,15, which corresponds to idx = 1,...,16,

    ```
    FE_k = ceil(10^(k/5 - 3) * MaxFES)
    ```
    
    An index value of 17 is interpreted as MaxFES.
    """

    maxfes = MAXFES_BY_D[D]
    points = []

    for k in range(16):
        fes = int(np.ceil((10.0 ** (k / 5.0 - 3.0)) * maxfes))
        fes = max(1, min(fes, maxfes))
        points.append(fes)

    return np.array(points, dtype=float)


def read_numeric_matrix(filepath):
    """
    Read a numeric matrix from a TXT or CSV file.

    Accepted formats:
    - space- or tab-separated values
    - comma-separated values
    - semicolon-separated values
    """
    filepath = Path(filepath)
    last_error = None

    for delimiter in (None, ",", ";", "\t"):
        try:
            data = np.loadtxt(filepath, dtype=float, delimiter=delimiter)
            return np.asarray(data, dtype=float)
        except Exception as e:
            last_error = e

    try:
        data = np.genfromtxt(filepath, dtype=float, delimiter=",")
        return np.asarray(data, dtype=float)
    except Exception:
        pass

    raise ValueError(
        f"Không đọc được file {filepath.name}. "
        f"File có thể không phải ma trận số hợp lệ. Lỗi gốc: {last_error}"
    )


def normalize_cec_matrix_shape(data, filepath, n_runs=N_RUNS):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"{filepath.name} không phải ma trận 2D. Shape hiện tại: {data.shape}"
        )

    if data.shape == (n_runs, 17):
        data = data.T

    if data.shape != (17, n_runs):
        raise ValueError(
            f"{filepath.name} sai kích thước {data.shape}. "
            f"CEC2022 yêu cầu 17 x {n_runs}."
        )

    return data


def convert_checkpoint_index_to_feterm(errors_matrix, feterm_raw, D, filepath):
    maxfes = MAXFES_BY_D[D]
    record_points = cec2022_record_points(D)
    feterm = np.empty_like(feterm_raw, dtype=float)

    for r, raw in enumerate(feterm_raw):
        if np.isfinite(raw):
            idx = int(round(float(raw)))
        else:
            idx = 0

        if 1 <= idx <= 16:
            feterm[r] = record_points[idx - 1]
        elif idx == 17:
            feterm[r] = maxfes
        else:
            hit_rows = np.where(errors_matrix[:, r] <= EPS)[0]
            if len(hit_rows) > 0:
                first_hit_row = int(hit_rows[0])
                feterm[r] = record_points[first_hit_row]
            else:
                feterm[r] = maxfes

    unique_raw = sorted(set(float(x) for x in feterm_raw if np.isfinite(x)))
    print(
        f"WARNING: {filepath.name} contains checkpoint indices in row 17 "
        f"{unique_raw}. These values have been automatically converted "
        f"to the corresponding FES values."
    )
    return feterm


def load_cec2022_result_file(filepath, D, n_runs=N_RUNS):
    """
    Read one CEC2022 result file.

    Returns:
        errors_final: final error from recording row k=15, shape (30,)
        feterm      : actual FEterm or converted FEterm, shape (30,)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Missing file: {filepath}")
    if D not in MAXFES_BY_D:
        raise ValueError(f"Chưa khai báo MaxFES cho D={D}")

    data = read_numeric_matrix(filepath)
    data = normalize_cec_matrix_shape(data, filepath, n_runs=n_runs)

    maxfes = MAXFES_BY_D[D]

    errors_matrix = data[:16, :].astype(float)
    feterm_raw = data[16, :].astype(float)

    # --------------------------------------------------------
    # Handle NaN/Inf values in the error matrix
    # --------------------------------------------------------
    bad_error_mask = ~np.isfinite(errors_matrix)

    if np.any(bad_error_mask):
        bad_pos = np.argwhere(bad_error_mask)
        bad_show = [(int(i) + 1, int(j) + 1) for i, j in bad_pos[:10]]
        print(
             f"WARNING: {filepath.name} contains NaN/Inf values in the error matrix "
             f"at positions such as {bad_show}. These values are treated as failed trials."
        )
        errors_matrix[bad_error_mask] = 1e300

    if np.any(errors_matrix < -1e-10):
        bad_pos = np.argwhere(errors_matrix < -1e-10)
        bad_show = [(int(i) + 1, int(j) + 1) for i, j in bad_pos[:10]]
        raise ValueError(
            f"{filepath.name} contains unusually negative error values "
            f"at positions such as {bad_show}."
        )

    errors_matrix = np.maximum(errors_matrix, 0.0)
    errors_final = errors_matrix[15, :].copy()

    bad_feterm_mask = ~np.isfinite(feterm_raw)
    finite_feterm = feterm_raw[np.isfinite(feterm_raw)]

    looks_like_checkpoint_index = (
        len(finite_feterm) > 0
        and np.nanmin(finite_feterm) >= 0
        and np.nanmax(finite_feterm) <= 17
        and np.all(np.isclose(finite_feterm, np.round(finite_feterm)))
    )

    if looks_like_checkpoint_index or np.any(bad_feterm_mask):
        feterm = convert_checkpoint_index_to_feterm(
            errors_matrix=errors_matrix,
            feterm_raw=np.where(np.isfinite(feterm_raw), feterm_raw, -1),
            D=D,
            filepath=filepath,
        )
    else:
        feterm = feterm_raw.copy()

        if np.any(feterm <= 0):
            bad = np.where(feterm <= 0)[0] + 1
            raise ValueError(
                 f"{filepath.name} contains FEterm <= 0 in runs {bad.tolist()}."
            )

        if np.any(feterm > maxfes):
            bad_idx = np.where(feterm > maxfes)[0]
            bad_runs = bad_idx + 1
            bad_values = feterm[bad_idx]
            print(
                 f"WARNING: {filepath.name} contains FEterm > MaxFES={maxfes} "
                 f"in runs {bad_runs.tolist()}, values = {bad_values.tolist()}. "
                 f"These values have been clipped to MaxFES."
            )
            feterm[bad_idx] = maxfes

    feterm = np.asarray(feterm, dtype=float)
    feterm = np.where(np.isfinite(feterm), feterm, maxfes)
    feterm = np.maximum(feterm, 1)
    feterm = np.minimum(feterm, maxfes)
    not_reached = errors_final > EPS
    feterm[not_reached] = maxfes

    return errors_final, feterm


# ============================================================
# CEC2022 COMPLIANT PAIRWISE SCORING
# ============================================================

def normalize_trial_for_cec2022(error, feterm, maxfes):
    """
    Normalize one trial according to the CEC2022 Technical Report criterion.

    Returns:
        reached   : True if error <= EPS
        comp_error: error value used for comparison when EPS is not reached
        comp_fe   : FEterm value used for comparison when EPS is reached

    Rules:
        - If error <= EPS, the trial is considered to have reached the target.
          Values such as 5e-9, 8e-9, and 1e-8 are no longer distinguished;
          the comparison is based on FEterm.
        - If error > EPS, the trial is considered unreached; FEterm is set
          to MaxFES, and the comparison is based on final error.
    """
    error = float(error)
    feterm = float(feterm)

    if not np.isfinite(error):
        return False, 1e300, float(maxfes)

    error = max(error, 0.0)

    if error <= EPS:
        if not np.isfinite(feterm) or feterm <= 0:
            feterm = maxfes
        feterm = min(max(float(feterm), 1.0), float(maxfes))
        return True, EPS, feterm

    return False, error, float(maxfes)


def pairwise_cec2022_tr_score(errors_a, feterm_a, errors_b, feterm_b, maxfes):
    """
    Compute the pairwise score between algorithms A and B according to
    the CEC2022 TR-compliant rule.

    errors_a, errors_b: shape (30,)
    feterm_a, feterm_b: shape (30,)

    Returns:
        score_a, score_b

    Rules:
        - reached vs. not reached: the reached trial wins.
        - reached vs. reached: the trial with the smaller FEterm wins.
        - not reached vs. not reached: the trial with the smaller final error wins.
        - exact tie: each algorithm receives 0.5 point.
     """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    feterm_a = np.asarray(feterm_a, dtype=float)
    feterm_b = np.asarray(feterm_b, dtype=float)

    reached_a = errors_a <= EPS
    reached_b = errors_b <= EPS

    fe_a = np.where(np.isfinite(feterm_a), feterm_a, maxfes).astype(float)
    fe_b = np.where(np.isfinite(feterm_b), feterm_b, maxfes).astype(float)
    fe_a = np.clip(fe_a, 1.0, float(maxfes))
    fe_b = np.clip(fe_b, 1.0, float(maxfes))

    err_a = np.nan_to_num(errors_a, nan=1e300, posinf=1e300, neginf=0.0)
    err_b = np.nan_to_num(errors_b, nan=1e300, posinf=1e300, neginf=0.0)
    err_a = np.maximum(err_a, 0.0)
    err_b = np.maximum(err_b, 0.0)

    # Broadcast 30 x 30 comparisons.
    ra = reached_a[:, None]
    rb = reached_b[None, :]
    fa = fe_a[:, None]
    fb = fe_b[None, :]
    ea = err_a[:, None]
    eb = err_b[None, :]

    # Case 1: A reached, B not reached.
    a_reached_only = ra & (~rb)
    b_reached_only = (~ra) & rb

    # Case 2: both reached -> compare FEterm.
    both_reached = ra & rb
    a_better_fe = both_reached & (fa < fb)
    b_better_fe = both_reached & (fa > fb)
    tie_fe = both_reached & (fa == fb)

    # Case 3: both not reached -> compare final error.
    both_not = (~ra) & (~rb)
    a_better_err = both_not & (ea < eb)
    b_better_err = both_not & (ea > eb)
    tie_err = both_not & (ea == eb)

    score_a = (
        np.sum(a_reached_only)
        + np.sum(a_better_fe)
        + np.sum(a_better_err)
        + 0.5 * (np.sum(tie_fe) + np.sum(tie_err))
    )

    score_b = (
        np.sum(b_reached_only)
        + np.sum(b_better_fe)
        + np.sum(b_better_err)
        + 0.5 * (np.sum(tie_fe) + np.sum(tie_err))
    )

    return float(score_a), float(score_b)


def calculate_cec2022_summary_report(
    algo_list,
    dimensions=(10, 20),
    num_functions=12,
    base_dir=".",
    strict=True,
    save_csv=True,
):
    """
    Compute the ranking score table according to the CEC2022 TR-compliant rule.

    Rules:
    - If error <= 1e-8, the trial is considered to have reached the target
      and is compared using FEterm.
    - If error > 1e-8, the trial is considered unreached and is compared
      using the final error.
    - Scores are computed directly from pairwise algorithm comparisons;
      for each dimension and function, the total score of each algorithm
      pair must be 30 * 30.
    """
    base_dir = Path(base_dir)
    warnings = []

    all_function_scores = []
    all_pairwise_rows = []
    all_trial_rows = []

    for D in dimensions:
        if D not in MAXFES_BY_D:
            raise ValueError(f"MaxFES has not been defined for D={D}")
        for fid in range(1, num_functions + 1):
            data_by_algo = {}
            loaded_algos = []

            # Load all algorithms for this D,F
            for algo_name in algo_list:
                filename = f"{algo_name}_{fid}_{D}.txt"
                filepath = base_dir / filename

                try:
                    errors, feterm = load_cec2022_result_file(
                        filepath=filepath,
                        D=D,
                        n_runs=N_RUNS,
                    )
                    data_by_algo[algo_name] = (errors, feterm)
                    loaded_algos.append(algo_name)

                    for run_id in range(N_RUNS):
                        all_trial_rows.append({
                            "Dimension": D,
                            "Function": fid,
                            "Algorithm": algo_name,
                            "Run": run_id + 1,
                            "FinalError": float(errors[run_id]),
                            "FEterm": float(feterm[run_id]),
                            "Reached": bool(errors[run_id] <= EPS),
                        })

                except Exception as e:
                    msg = f"[D={D}, F{fid}, {algo_name}] {e}"
                    if strict:
                        raise RuntimeError(msg) from e
                    warnings.append(msg)
                    continue

            if strict and len(loaded_algos) != len(algo_list):
                missing = sorted(set(algo_list) - set(loaded_algos))
                raise RuntimeError(
                    f"Missing algorithms for D={D}, F{fid}: {missing}"
                )

            if len(loaded_algos) < 2:
                if strict:
                    raise RuntimeError(f"Insufficient data to rank algorithms for D={D}, F{fid}")
                continue

            scores_this_case = {algo: 0.0 for algo in loaded_algos}

            # Pairwise scoring theo CEC2022 TR: reached EPS -> compare FEterm.
            for ia in range(len(loaded_algos)):
                algo_a = loaded_algos[ia]
                err_a, fe_a = data_by_algo[algo_a]

                for ib in range(ia + 1, len(loaded_algos)):
                    algo_b = loaded_algos[ib]
                    err_b, fe_b = data_by_algo[algo_b]

                    score_a, score_b = pairwise_cec2022_tr_score(
                        err_a, fe_a, err_b, fe_b, maxfes=MAXFES_BY_D[D]
                    )

                    scores_this_case[algo_a] += score_a
                    scores_this_case[algo_b] += score_b

                    all_pairwise_rows.append({
                        "Dimension": D,
                        "Function": fid,
                        "Algorithm_A": algo_a,
                        "Algorithm_B": algo_b,
                        "Score_A": score_a,
                        "Score_B": score_b,
                        "Score_Sum": score_a + score_b,
                    })

            expected_pair_sum = (N_RUNS ** 2) * len(loaded_algos) * (len(loaded_algos) - 1) / 2.0
            actual_pair_sum = sum(scores_this_case.values())
            if abs(actual_pair_sum - expected_pair_sum) > 1e-6:
                raise RuntimeError(
                    f"D={D}, F{fid}: tổng điểm pairwise sai. "
                    f"Actual={actual_pair_sum}, Expected={expected_pair_sum}"
                )

            for algo_name, score in scores_this_case.items():
                all_function_scores.append({
                    "Dimension": D,
                    "Function": fid,
                    "Algorithm": algo_name,
                    "Score": float(score),
                })

    if len(all_function_scores) == 0:
        raise RuntimeError("No valid data are available for score calculation.")

    df_function_scores = pd.DataFrame(all_function_scores)
    df_pairwise_scores = pd.DataFrame(all_pairwise_rows)
    df_all_trials = pd.DataFrame(all_trial_rows)

    summary = (
        df_function_scores
        .groupby(["Algorithm", "Dimension"], as_index=False)["Score"]
        .sum()
    )

    summary = summary.pivot(
        index="Algorithm",
        columns="Dimension",
        values="Score",
    ).fillna(0.0)

    for D in dimensions:
        if D not in summary.columns:
            summary[D] = 0.0

    summary = summary[list(dimensions)]
    summary.columns = [f"D={D}" for D in dimensions]

    summary["Total Score"] = summary.sum(axis=1)
    summary["Overall Rank"] = (
        summary["Total Score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    summary = summary.sort_values(
        by=["Overall Rank", "Total Score"],
        ascending=[True, False],
    ).reset_index()

    ordered_cols = (
        ["Overall Rank", "Algorithm"]
        + [f"D={D}" for D in dimensions]
        + ["Total Score"]
    )
    summary = summary[ordered_cols]

    if save_csv:
        summary.to_csv(base_dir / "CEC2022_Overall_Summary_Report.csv", index=False)
        df_function_scores.to_csv(base_dir / "CEC2022_Function_Scores.csv", index=False)
        df_pairwise_scores.to_csv(base_dir / "CEC2022_Pairwise_Scores.csv", index=False)
        df_all_trials.to_csv(base_dir / "CEC2022_All_Trial_Data.csv", index=False)

    if warnings:
        print("\nWARNINGS when strict=False:")
        for w in warnings:
            print(" -", w)

    return summary, df_function_scores, df_pairwise_scores, df_all_trials


# ============================================================
# OUTPUT HELPERS
# ============================================================

def check_score_sum(summary_report, dimensions=(10, 20), n_runs=N_RUNS, n_functions=12):
    """
    Check the theoretical total score.

    For m algorithms, the total score for each dimension must be:
        n_functions * n_runs^2 * m * (m - 1) / 2
    """
    m = len(summary_report)

    print("\n" + "=" * 80)
    print("CHECKING THEORETICAL TOTAL SCORES")
    print("=" * 80)

    for D in dimensions:
        col = f"D={D}"
        actual = float(summary_report[col].sum())
        expected = n_functions * (n_runs ** 2) * m * (m - 1) / 2.0
        diff = actual - expected

        print(f"{col}:")
        print(f"  Actual total score      = {actual:.1f}")
        print(f"  Theoretical total score = {expected:.1f}")
        print(f"  Difference              = {diff:.6f}")

    print("=" * 80)


def check_pairwise_sum(pairwise_scores, n_runs=N_RUNS, n_functions=12):
    if pairwise_scores.empty:
        return
    bad = pairwise_scores[np.abs(pairwise_scores["Score_Sum"] - n_runs ** 2) > 1e-6]

    print("\n" + "=" * 80)
    print("CHECKING PAIRWISE TOTAL SCORES")
    print("=" * 80)

    if len(bad) == 0:
        print(
              f"All algorithm pairs have Score_A + Score_B = {n_runs**2} "
              f"for each D,F case."
        )
    else:
        print("Some pairs have incorrect pairwise total scores:")
        print(bad.head(20).to_string(index=False))
    print("=" * 80)


def print_summary_table(summary_report):
    print("\n" + "=" * 105)
    print("OVERALL ALGORITHM RANKING SUMMARY - CEC2022 TR/EPS-FETERM")
    print("=" * 105)

    dim_cols = [c for c in summary_report.columns if c.startswith("D=")]

    header = f"{'Rank':<6} {'Algorithm':<25}"
    for c in dim_cols:
        header += f" {c:<18}"
    header += f" {'Total Score':<18}"
    print(header)
    print("-" * 105)

    for _, row in summary_report.iterrows():
        line = f"{int(row['Overall Rank']):<6} {row['Algorithm']:<25}"
        for c in dim_cols:
            line += f" {row[c]:<18.1f}"
        line += f" {row['Total Score']:<18.1f}"
        print(line)

    print("=" * 105)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    my_algos = [
        "PDABC",
         "ABC",
         "ZOCMAES",
        "EA4eigN100_10",
        "NL-SHADE-LBC",
        "S_LSHADE_DP",
        "NL-SHADE-RSP-MID",
        "Co-PPSO",
        "NLSOMACLP",
    ]

    summary_report, function_scores, pairwise_scores, all_trial_data = calculate_cec2022_summary_report(
        algo_list=my_algos,
        dimensions=(10, 20),
        num_functions=12,
        base_dir=RESULTS_DIR,
        strict=True,
        save_csv=True,
    )

    print_summary_table(summary_report)
    check_score_sum(summary_report, dimensions=(10, 20), n_functions=12)
    check_pairwise_sum(pairwise_scores)

    print("\nSaved:")
    print(f" - {RESULTS_DIR / 'CEC2022_Overall_Summary_Report.csv'}")
    print(f" - {RESULTS_DIR / 'CEC2022_Function_Scores.csv'}")
    print(f" - {RESULTS_DIR / 'CEC2022_Pairwise_Scores.csv'}")
    print(f" - {RESULTS_DIR / 'CEC2022_All_Trial_Data.csv'}")
