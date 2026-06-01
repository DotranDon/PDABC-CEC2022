import argparse
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, friedmanchisquare, rankdata, wilcoxon

# ============================================================
# PDABC WILCOXON / FRIEDMAN STATISTICAL ANALYSIS
# CEC2022 TR / EPS-FEterm compliant version
#
# Default input :CEC2022_All_Trial_Data.csv   (from cec2022_target_reaching_score.py)
#
# Required columns:
#   Dimension, Function, Algorithm, Run, FinalError, FEterm
#
# Trial value used for all statistical tests:
#   if FinalError <= 1e-8: TrialValue_TR = FEterm
#   else:                  TrialValue_TR = 1e15 + FinalError
#
# Smaller TrialValue_TR is better.
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_INPUT = RESULTS_DIR / "CEC2022_All_Trial_Data.csv"
DEFAULT_CONTROL_ALGO = "PDABC"
VALUE_COL = "TrialValue_TR"
EPS = 1e-8
BIG = 1e15


# ============================================================
# INPUT RESOLUTION / LOAD DATA
# ============================================================

def resolve_input_path(user_input=None):
    if user_input:
        p = Path(user_input)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found " )
        return p
    p = Path(DEFAULT_INPUT)
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Input file not found. "
    )


def load_data(path, control_algo):
    df = pd.read_csv(path)

    required = ["Dimension", "Function", "Algorithm", "Run", "FinalError", "FEterm"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()
    df["Algorithm"] = df["Algorithm"].astype(str).str.strip()
    df["Dimension"] = df["Dimension"].astype(int)
    df["Function"] = df["Function"].astype(int)
    df["Run"] = df["Run"].astype(int)
    df["FinalError"] = pd.to_numeric(df["FinalError"], errors="coerce")
    df["FEterm"] = pd.to_numeric(df["FEterm"], errors="coerce")

    # Sanitize values.
    df["FinalError"] = df["FinalError"].replace([np.inf, -np.inf], np.nan)
    df["FinalError"] = df["FinalError"].fillna(BIG)
    df["FinalError"] = np.maximum(df["FinalError"].to_numpy(dtype=float), 0.0)

    df["FEterm"] = df["FEterm"].replace([np.inf, -np.inf], np.nan)
    # FEterm is only used for reached trials. If missing, set to a large value.
    df["FEterm"] = df["FEterm"].fillna(BIG)
    df["FEterm"] = np.maximum(df["FEterm"].to_numpy(dtype=float), 1.0)

    # IMPORTANT: always rebuild TrialValue_TR according to CEC2022 TR/EPS-FEterm rule.
    # Do not trust an existing TrialValue column from another scoring script.
    reached = df["FinalError"].to_numpy(dtype=float) <= EPS
    df["ReachedTR"] = reached
    df[VALUE_COL] = np.where(
        reached,
        df["FEterm"].to_numpy(dtype=float),
        BIG + df["FinalError"].to_numpy(dtype=float),
    )

    algos = sorted(df["Algorithm"].unique())
    if control_algo not in algos:
        raise ValueError(
            f"Control algorithm {control_algo} not found in file {path}.\n"
            f"Available algorithms: {algos}\n"
        )

    return df


# ============================================================
# WILCOXON RANK-SUM / MANN-WHITNEY U PAIRWISE TEST
# ============================================================

def classify_pair(control_vals, other_vals, p_value, alpha):
    """
    Return:
      + : control significantly better
      = : no significant difference
      - : control significantly worse

    Smaller value is better.
    """
    if not np.isfinite(p_value) or p_value >= alpha:
        return "="

    c_mean = float(np.mean(control_vals))
    o_mean = float(np.mean(other_vals))

    if c_mean < o_mean:
        return "+"
    if c_mean > o_mean:
        return "-"
    return "="


def compute_wilcoxon_details(df, control_algo, alpha):
    algorithms = sorted(df["Algorithm"].unique())
    competitors = [a for a in algorithms if a != control_algo]

    rows = []

    cases = (
        df[["Dimension", "Function"]]
        .drop_duplicates()
        .sort_values(["Dimension", "Function"])
        .itertuples(index=False)
    )

    for D, fid in cases:
        case_df = df[(df["Dimension"] == D) & (df["Function"] == fid)]

        control_vals = (
            case_df[case_df["Algorithm"] == control_algo]
            .sort_values("Run")[VALUE_COL]
            .to_numpy(dtype=float)
        )

        if len(control_vals) == 0:
            raise ValueError(f"No {control_algo} data for D={D}, F={fid}")

        for algo in competitors:
            other_vals = (
                case_df[case_df["Algorithm"] == algo]
                .sort_values("Run")[VALUE_COL]
                .to_numpy(dtype=float)
            )

            if len(other_vals) == 0:
                continue

            if len(control_vals) == len(other_vals) and np.array_equal(control_vals, other_vals):
                stat = np.nan
                p_value = 1.0
            else:
                try:
                    stat, p_value = mannwhitneyu(
                        control_vals,
                        other_vals,
                        alternative="two-sided",
                        method="auto",
                    )
                except Exception:
                    stat = np.nan
                    p_value = 1.0

            relation = classify_pair(control_vals, other_vals, p_value, alpha)

            rows.append({
                "Algorithm": algo,
                "Dimension": int(D),
                "Function": int(fid),
                "Control": control_algo,
                "Control_mean": float(np.mean(control_vals)),
                "Other_mean": float(np.mean(other_vals)),
                "Control_median": float(np.median(control_vals)),
                "Other_median": float(np.median(other_vals)),
                "Statistic": stat,
                "p_value": p_value,
                "Relation": relation,
            })

    return pd.DataFrame(rows)


def summarize_wilcoxon(details):
    rows = []

    for algo in sorted(details["Algorithm"].unique()):
        sub = details[details["Algorithm"] == algo]

        total_plus = int((sub["Relation"] == "+").sum())
        total_equal = int((sub["Relation"] == "=").sum())
        total_minus = int((sub["Relation"] == "-").sum())

        row = {
            "Algorithm": algo,
            "Total_+": total_plus,
            "Total_=": total_equal,
            "Total_-": total_minus,
            "Total_+/=/-": f"{total_plus}/{total_equal}/{total_minus}",
        }

        for D in sorted(details["Dimension"].unique()):
            dsub = sub[sub["Dimension"] == D]

            p = int((dsub["Relation"] == "+").sum())
            e = int((dsub["Relation"] == "=").sum())
            m = int((dsub["Relation"] == "-").sum())

            row[f"D={D}_+"] = p
            row[f"D={D}_="] = e
            row[f"D={D}_-"] = m
            row[f"D={D}_+/=/-"] = f"{p}/{e}/{m}"

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        by=["Total_+", "Total_-", "Total_="],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    return summary


# ============================================================
# FRIEDMAN TEST AND AVERAGE RANKS
# ============================================================

def compute_case_mean_matrix(df, control_algo):
    """
    Matrix:
      rows    = cases (Dimension, Function)
      columns = algorithms
      value   = mean TrialValue_TR over 30 runs
    """
    case_mean = (
        df.groupby(["Dimension", "Function", "Algorithm"], as_index=False)[VALUE_COL]
        .mean()
        .rename(columns={VALUE_COL: "MeanTrialValue_TR"})
    )

    pivot = case_mean.pivot_table(
        index=["Dimension", "Function"],
        columns="Algorithm",
        values="MeanTrialValue_TR",
    )

    pivot = pivot.sort_index().dropna(axis=0, how="any")
    algorithms = list(pivot.columns)

    if control_algo not in algorithms:
        raise ValueError(f"{control_algo} is missing from Friedman matrix.")

    return pivot, algorithms


def compute_friedman(df, control_algo):
    pivot, algorithms = compute_case_mean_matrix(df, control_algo)
    matrix = pivot.to_numpy(dtype=float)

    stat, p_value = friedmanchisquare(
        *[matrix[:, j] for j in range(matrix.shape[1])]
    )

    rank_matrix = np.vstack([rankdata(row, method="average") for row in matrix])
    avg_ranks = rank_matrix.mean(axis=0)

    friedman_ranks = pd.DataFrame({
        "Algorithm": algorithms,
        "AverageRank": avg_ranks,
    }).sort_values("AverageRank").reset_index(drop=True)

    friedman_ranks.insert(0, "Rank", np.arange(1, len(friedman_ranks) + 1))

    case_rows = []
    for case_idx, ((D, fid), row_values) in enumerate(zip(pivot.index, matrix)):
        ranks = rank_matrix[case_idx]
        for algo, mean_val, rank_val in zip(algorithms, row_values, ranks):
            case_rows.append({
                "Dimension": int(D),
                "Function": int(fid),
                "Algorithm": algo,
                "MeanTrialValue_TR": float(mean_val),
                "CaseRank": float(rank_val),
            })

    case_ranks = pd.DataFrame(case_rows).sort_values(
        ["Dimension", "Function", "CaseRank", "Algorithm"]
    ).reset_index(drop=True)

    friedman_info = {
        "statistic": float(stat),
        "p_value": float(p_value),
        "n_cases": int(matrix.shape[0]),
        "n_algorithms": int(matrix.shape[1]),
        "algorithms": algorithms,
        "rank_matrix": rank_matrix,
        "pivot": pivot,
    }

    return friedman_info, friedman_ranks, case_ranks


# ============================================================
# POST-HOC WILCOXON ON CASE RANKS + HOLM CORRECTION
# ============================================================

def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0

    for i, p in enumerate(sorted_p):
        adj = min((m - i) * p, 1.0)
        running_max = max(running_max, adj)
        adjusted_sorted[i] = running_max

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def compute_posthoc_vs_control(friedman_info, friedman_ranks, control_algo, alpha):
    algorithms = friedman_info["algorithms"]
    rank_matrix = friedman_info["rank_matrix"]

    control_idx = algorithms.index(control_algo)
    control_ranks = rank_matrix[:, control_idx]

    rows = []

    for algo in algorithms:
        if algo == control_algo:
            continue

        algo_idx = algorithms.index(algo)
        algo_ranks = rank_matrix[:, algo_idx]
        diffs = control_ranks - algo_ranks

        if np.allclose(diffs, 0):
            stat = np.nan
            p_value = 1.0
        else:
            try:
                stat, p_value = wilcoxon(
                    control_ranks,
                    algo_ranks,
                    alternative="two-sided",
                    zero_method="wilcox",
                    mode="auto",
                )
            except Exception:
                stat = np.nan
                p_value = 1.0

        control_avg = float(
            friedman_ranks.loc[
                friedman_ranks["Algorithm"] == control_algo,
                "AverageRank",
            ].iloc[0]
        )
        algo_avg = float(
            friedman_ranks.loc[
                friedman_ranks["Algorithm"] == algo,
                "AverageRank",
            ].iloc[0]
        )

        rows.append({
            "Control": control_algo,
            "Algorithm": algo,
            "ControlAverageRank": control_avg,
            "AlgorithmAverageRank": algo_avg,
            "AverageRankDifference_AlgorithmMinusControl": algo_avg - control_avg,
            "Statistic": stat,
            "p_value": p_value,
        })

    posthoc = pd.DataFrame(rows)
    posthoc["Holm_p"] = holm_adjust(posthoc["p_value"].to_numpy(dtype=float))
    posthoc["Significant_Holm_0.05"] = posthoc["Holm_p"] < alpha
    return posthoc.sort_values("p_value").reset_index(drop=True)


# ============================================================
# REPORT / OUTPUTS
# ============================================================

def write_report(
    output_prefix,
    input_path,
    control_algo,
    alpha,
    wilcoxon_summary,
    friedman_info,
    friedman_ranks,
    posthoc,
):
    lines = []
    lines.append("PDABC Wilcoxon / Friedman Statistical Report")
    lines.append("CEC2022 TR / EPS-FEterm scoring")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Input file        : {input_path}")
    lines.append(f"Control algorithm : {control_algo}")
    lines.append(f"Value column      : {VALUE_COL}")
    lines.append(f"Alpha             : {alpha}")
    lines.append("")
    lines.append("TrialValue_TR rule:")
    lines.append("  if FinalError <= 1e-8: TrialValue_TR = FEterm")
    lines.append("  else:                  TrialValue_TR = 1e15 + FinalError")
    lines.append("  Smaller TrialValue_TR is better.")
    lines.append("")

    lines.append("Wilcoxon Rank-Sum Summary (+/=/- over 24 cases)")
    lines.append("-" * 70)
    for _, row in wilcoxon_summary.iterrows():
        lines.append(
            f"{row['Algorithm']:<20} "
            f"Total {row['Total_+/=/-']:<8} "
            + " ".join([
                f"{col.replace('_+/=/-', '')}: {row[col]}"
                for col in wilcoxon_summary.columns
                if col.startswith("D=") and col.endswith("_+/=/-")
            ])
        )

    lines.append("")
    lines.append("Friedman Test")
    lines.append("-" * 70)
    lines.append(f"Number of cases      : {friedman_info['n_cases']}")
    lines.append(f"Number of algorithms : {friedman_info['n_algorithms']}")
    lines.append(f"Chi-square statistic : {friedman_info['statistic']:.6f}")
    lines.append(f"p-value              : {friedman_info['p_value']:.8f}")
    lines.append("")

    lines.append("Average Friedman Ranks")
    lines.append("-" * 70)
    for _, row in friedman_ranks.iterrows():
        lines.append(
            f"{int(row['Rank']):>2}. {row['Algorithm']:<20} "
            f"{row['AverageRank']:.6f}"
        )

    lines.append("")
    lines.append(f"Post-hoc Wilcoxon on Case Ranks vs {control_algo} with Holm Correction")
    lines.append("-" * 70)
    for _, row in posthoc.iterrows():
        sig = "YES" if row["Significant_Holm_0.05"] else "NO"
        lines.append(
            f"{row['Algorithm']:<20} "
            f"p={row['p_value']:.6f} "
            f"Holm_p={row['Holm_p']:.6f} "
            f"Significant={sig}"
        )

    report_text = "\n".join(lines)
    report_path = RESULTS_DIR / f"{output_prefix}_Stats_Report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    return report_text, report_path


def zip_outputs(output_prefix, files):
    zip_path = RESULTS_DIR / f"{output_prefix}_Wilcoxon_Friedman_Stats.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in files:
            p = Path(filename)
            if p.exists():
                zf.write(p, arcname=p.name)
    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="PDABC Wilcoxon/Friedman stats under CEC2022 TR/EPS-FEterm scoring."
    )
    parser.add_argument("--input", type=str, default=None, help="Input CSV file.")
    parser.add_argument("--control", type=str, default=DEFAULT_CONTROL_ALGO, help="Control algorithm name.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument("--out_prefix", type=str, default="PDABC_TR", help="Output file prefix.")
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    control_algo = args.control.strip()
    alpha = float(args.alpha)
    prefix = args.out_prefix

    print(f"Loading data from {input_path}...")
    df = load_data(input_path, control_algo)

    print("Algorithms found:")
    print(", ".join(sorted(df["Algorithm"].unique())))

    print("Computing Wilcoxon rank-sum tests...")
    wilcoxon_details = compute_wilcoxon_details(df, control_algo, alpha)
    wilcoxon_summary = summarize_wilcoxon(wilcoxon_details)

    print("Computing Friedman test and average ranks...")
    friedman_info, friedman_ranks, case_ranks = compute_friedman(df, control_algo)

    print(f"Computing post-hoc tests vs {control_algo}...")
    posthoc = compute_posthoc_vs_control(friedman_info, friedman_ranks, control_algo, alpha)

    output_files = []
    paths = {
        "wilcoxon_summary": RESULTS_DIR / f"{prefix}_Wilcoxon_Summary.csv",
        "wilcoxon_details": RESULTS_DIR / f"{prefix}_Wilcoxon_Details.csv",
        "friedman_ranks": RESULTS_DIR / f"{prefix}_Friedman_Ranks.csv",
        "friedman_posthoc": RESULTS_DIR / f"{prefix}_Friedman_Posthoc.csv",
        "case_ranks": RESULTS_DIR / f"{prefix}_Case_Ranks.csv",
    }

    print("Saving CSV files...")
    wilcoxon_summary.to_csv(paths["wilcoxon_summary"], index=False)
    wilcoxon_details.to_csv(paths["wilcoxon_details"], index=False)
    friedman_ranks.to_csv(paths["friedman_ranks"], index=False)
    posthoc.to_csv(paths["friedman_posthoc"], index=False)
    case_ranks.to_csv(paths["case_ranks"], index=False)
    output_files.extend(paths.values())

    print("Writing text report...")
    report_text, report_path = write_report(
        output_prefix=prefix,
        input_path=input_path,
        control_algo=control_algo,
        alpha=alpha,
        wilcoxon_summary=wilcoxon_summary,
        friedman_info=friedman_info,
        friedman_ranks=friedman_ranks,
        posthoc=posthoc,
    )
    output_files.append(report_path)

    print("Creating zip archive...")
    zip_path = zip_outputs(prefix, output_files)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(report_text)
    print("\nCreated files:")
    for p in output_files:
        print(f" - {p}")
    print(f" - {zip_path}")


if __name__ == "__main__":
    main()
