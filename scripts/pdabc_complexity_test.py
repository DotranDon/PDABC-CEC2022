import numpy as np
import time
import platform
import opfunu
from pathlib import Path
from dataclasses import dataclass


# ============================================================
# CEC2022 ALGORITHM COMPLEXITY TEST FOR PD-ABC 
#
# T0:
#   CEC2022 basic operation test program, 200000 iterations.
#
# T1:
#   Time for 200000 evaluations of Function 1.
#
# T2:
#   Time for the complete PDABC reduced2 algorithm with 200000
#   function evaluations on Function 1.
#   T2 is repeated five times and averaged.
#
# Complexity:
#   (T2 - T1) / T0
#
# T0 is measured once and reused for D = 10 and D = 20.
# ============================================================

EPS = 1e-8
CEC2022_F1_BIAS = 300.0
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class ABCConfig:
    D: int
    SN: int
    FE_max: int
    limit: int
    milestones: np.ndarray = None
    actual_seed: int = 0


# ============================================================
# Helper functions, matched to pdabc.py
# ============================================================

def ensure_bounds(x, lo=-100.0, hi=100.0):
    """
    Reflection + clipping boundary handling.
    This matches the boundary handling in pdabc.py.
    """
    x_new = x.copy()

    over_hi = x_new > hi
    x_new[over_hi] = 2.0 * hi - x_new[over_hi]

    under_lo = x_new < lo
    x_new[under_lo] = 2.0 * lo - x_new[under_lo]

    return np.clip(x_new, lo, hi)


def fit_transform(fvals):
    """ABC fitness transform for minimization."""
    fvals = np.asarray(fvals, dtype=float)
    return np.where(fvals >= 0.0, 1.0 / (1.0 + fvals), 1.0 + np.abs(fvals))


# ============================================================
# PDABC core for T2
# ============================================================

def PD_ABC_Core_Complexity(func, cfg: ABCConfig):
    """
    PDABC score for measuring T2.

    Important for CEC2022 complexity measurement:
    - The algorithm is run on F1 with exactly FE_max evaluations.
    - There is no early stopping at EPS here because the CEC2022
      complexity test defines T2 as the complete algorithm time for
      200000 evaluations on Function 1.
    - The search logic matches the latest TR-compliant PDABC reduced2 core:
        * p_j = S / sum(S)
        * lambda = FE / MaxFE
        * S update = log1p(relative improvement)
        * onlooker uses smaller K_on and shrinking Gaussian perturbation
        * scout preserves dimensions with S_j > median(S)
        * memory decay = 1 - 1/SN toward neutral baseline S_j = 1
    """
    D, SN, FE_max = cfg.D, cfg.SN, cfg.FE_max
    rng = np.random.default_rng(cfg.actual_seed)

    def init_population(n):
        return -100.0 + 200.0 * rng.random((n, D))

    def eval_one(x):
        val = float(func(x))
        if not np.isfinite(val):
            return np.inf
        return max(val, 0.0)

    def memory_gain(old_val, new_val):
        old_val = float(old_val)
        new_val = float(new_val)

        if np.isfinite(old_val) and np.isfinite(new_val):
            rel_imp = max((old_val - new_val) / (abs(old_val) + 1e-12), 0.0)
            return float(np.log1p(rel_imp))

        if np.isfinite(new_val):
            return 1.0

        return 0.0

    # Initialization
    X = init_population(SN)
    f = np.array([eval_one(X[i]) for i in range(SN)], dtype=float)
    fe = SN

    trial = np.zeros(SN, dtype=float)
    S = np.ones(D, dtype=float)

    best_idx = int(np.argmin(f))
    f_best = float(f[best_idx])
    x_best = X[best_idx].copy()

    # Main loop: no EPS early termination for complexity test.
    while fe < FE_max:
        progress = fe / FE_max if FE_max > 0 else 0.0
        K_floor = max(1, int((D // 2) * (1.0 - progress ** 2)))
        lam = progress
        S = np.nan_to_num(S, nan=1.0, posinf=1.0, neginf=1.0)
        S = np.maximum(S, 1.0)
        p_j = S / (np.sum(S) + 1e-12)

        # =====================================================
        # Phase 1: Employed bees
        # =====================================================
        for i in range(SN):
            if fe >= FE_max:
                break

            K = int(rng.integers(K_floor, D + 1))
            dims = rng.choice(D, size=K, p=p_j, replace=False)

            k = (i + int(rng.integers(1, SN))) % SN
            v = X[i].copy()

            phi = rng.uniform(-1.0, 1.0, size=K)
            r_best = rng.uniform(0.0, 1.0, size=K)

            v[dims] += (
                phi * (X[i, dims] - X[k, dims])
                + lam * r_best * (x_best[dims] - X[i, dims])
            )

            v = ensure_bounds(v)
            fv = eval_one(v)
            fe += 1

            if fv < f[i]:
                gain = memory_gain(f[i], fv)
                S[dims] += gain

                X[i] = v
                f[i] = fv
                trial[i] = 0

                if fv < f_best:
                    f_best = float(fv)
                    x_best = v.copy()
            else:
                trial[i] += 1

        if fe >= FE_max:
            break

        # =====================================================
        # Phase 2: Onlooker bees
        # =====================================================
        fit_raw = fit_transform(f)
        fit_raw = np.nan_to_num(fit_raw, nan=0.0, posinf=0.0, neginf=0.0)
        sum_fit = float(np.sum(fit_raw))

        if sum_fit > 1e-12 and np.isfinite(sum_fit):
            fit_p = fit_raw / sum_fit
        else:
            fit_p = np.ones(SN, dtype=float) / SN

        onlookers = rng.choice(SN, size=SN, p=fit_p, replace=True)

        for i in onlookers:
            if fe >= FE_max:
                break

            i = int(i)

            # Same K_on as PDABC: tied to K_floor, no independent parameter.
            high = max(2, K_floor // 2 + 1)
            K_on = int(rng.integers(1, high))
            K_on = max(1, min(K_on, D))

            dims = rng.choice(D, size=K_on, p=p_j, replace=False)

            k = (i + int(rng.integers(1, SN))) % SN
            v = X[i].copy()

            # Same focused refinement as reduced2.
            scale = (1.0 - progress) ** 2
            step = rng.standard_normal(size=K_on) * scale
            r_best = rng.uniform(0.0, 1.0, size=K_on)

            v[dims] = (
                X[i, dims]
                + step * (X[i, dims] - X[k, dims])
                + lam * r_best * (x_best[dims] - X[i, dims])
            )

            v = ensure_bounds(v)
            fv = eval_one(v)
            fe += 1

            if fv < f[i]:
                gain = memory_gain(f[i], fv)
                S[dims] += gain

                X[i] = v
                f[i] = fv
                trial[i] = 0

                if fv < f_best:
                    f_best = float(fv)
                    x_best = v.copy()
            else:
                trial[i] += 1

        if fe >= FE_max:
            break

        # =====================================================
        # Phase 3: Scout bees
        # =====================================================
        scouts = np.where(trial > cfg.limit)[0]

        if len(scouts) > 0:
            S_clean = np.nan_to_num(S, nan=1.0, posinf=1.0, neginf=1.0)
            S_clean = np.maximum(S_clean, 1.0)

            # Same median-based elite dimension preservation as PDABC.
            elite_mask = S_clean > np.median(S_clean)

            if not np.any(elite_mask):
                elite_mask[int(np.argmax(S_clean))] = True

            for i in scouts:
                if fe >= FE_max:
                    break

                new_bee = init_population(1)[0]
                X[i] = np.where(elite_mask, x_best, new_bee)

                f[i] = eval_one(X[i])
                fe += 1
                trial[i] = 0

                if f[i] < f_best:
                    f_best = float(f[i])
                    x_best = X[i].copy()

        # Same parameter-free memory decay toward neutral baseline S_j = 1.
        decay = 1.0 - 1.0 / SN
        S = 1.0 + (S - 1.0) * decay
        S = np.maximum(S, 1.0)

    if fe != FE_max:
        raise RuntimeError(f"FE counting error: fe={fe}, expected FE_max={FE_max}")

    return f_best


# ============================================================
# Measure T0, T1, T2
# ============================================================

def measure_T0(repeats=5):
    """
    Measure T0 using the CEC2022 basic operation test program.
    NumPy is used to avoid math-domain failures when log receives a
    non-positive intermediate value due to floating-point behavior.
    """
    times = []
    old_settings = np.seterr(divide="ignore", invalid="ignore", over="ignore", under="ignore")

    try:
        for _ in range(repeats):
            x = np.float64(0.55)

            start_time = time.perf_counter()

            for _ in range(200000):
                x = x + x
                x = x / 2.0
                x = x * x
                x = np.sqrt(x)
                x = np.log(x)
                x = np.exp(x)
                x = x / (x + 2.0)

            elapsed = time.perf_counter() - start_time
            times.append(elapsed)

    finally:
        np.seterr(**old_settings)

    return float(np.mean(times))


def measure_T1(problem_f1, D, max_evals=200000, seed=12345):
    """
    Measure T1: time for max_evals evaluations of Function 1.
    Since the algorithm evaluates one vector at a time, T1 is measured
    using the same one-vector-at-a-time style.
    """
    rng = np.random.default_rng(seed)
    sample_pop = rng.uniform(-100.0, 100.0, size=(max_evals, D))

    start_time = time.perf_counter()

    for i in range(max_evals):
        problem_f1.evaluate(sample_pop[i])

    return float(time.perf_counter() - start_time)


def measure_T2(problem_f1, D, max_evals=200000, repeats=5, base_seed=20260429):
    """
    Measure T2: run the full PDABC algorithm on Function 1
    for exactly max_evals evaluations. Repeat five times and average.
    """
    t2_runs = []

    def obj_func(x):
        # Match the benchmark runner: optimize the error value F(x) - F*.
        # This keeps the algorithmic path consistent with the PDABC
        # result-generation code while still measuring the same function.
        err = float(problem_f1.evaluate(x)) - CEC2022_F1_BIAS
        return max(err, 0.0)

    for j in range(repeats):
        cfg = ABCConfig(
            D=D,
            SN=30,
            FE_max=max_evals,
            limit=int(0.5 * 30 * D),
            actual_seed=base_seed + j,
        )

        start_time = time.perf_counter()
        PD_ABC_Core_Complexity(obj_func, cfg)
        elapsed = time.perf_counter() - start_time

        t2_runs.append(elapsed)

    return float(np.mean(t2_runs)), t2_runs


def run_complexity_test(D, t0):
    """
    Run T1 and T2 for one dimension. T0 is measured once outside this
    function and reused for both D = 10 and D = 20.
    """
    print(f"\n>>> Measuring computational complexity for D = {D}...")

    max_evals = 200000
    problem_f1 = opfunu.cec_based.cec2022.F12022(ndim=D)

    t1 = measure_T1(
        problem_f1=problem_f1,
        D=D,
        max_evals=max_evals,
        seed=1000 + D,
    )

    t2, t2_runs = measure_T2(
        problem_f1=problem_f1,
        D=D,
        max_evals=max_evals,
        repeats=5,
        base_seed=20260429 + D,
    )

    complexity = (t2 - t1) / t0

    return {
        "D": D,
        "T0": t0,
        "T1": t1,
        "T2": t2,
        "Complexity": complexity,
        "T2_runs": t2_runs,
    }


# ============================================================
# Output
# ============================================================

def print_complexity_table(results):
    print("\n" + "=" * 80)
    print("PDABC COMPUTATIONAL COMPLEXITY SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'D':<5} | "
        f"{'T0 (s)':<12} | "
        f"{'T1 (s)':<12} | "
        f"{'T2 mean (s)':<14} | "
        f"{'(T2-T1)/T0':<14}"
    )
    print("-" * 80)

    for res in results:
        print(
            f"{res['D']:<5} | "
            f"{res['T0']:<12.4f} | "
            f"{res['T1']:<12.4f} | "
            f"{res['T2']:<14.4f} | "
            f"{res['Complexity']:<14.4f}"
        )

    print("=" * 80)

    print("\nDetails of the 5 T2 runs:")
    for res in results:
        t2_str = ", ".join(f"{x:.4f}" for x in res["T2_runs"])
        print(f"D = {res['D']}: {t2_str}")


def print_system_info():
    print("\n" + "=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    print(f"Python     : {platform.python_version()}")
    print(f"Platform   : {platform.platform()}")
    print(f"Processor  : {platform.processor()}")
    print(f"NumPy      : {np.__version__}")
    print("=" * 80)


if __name__ == "__main__":
    print_system_info()

    print("\n>>> Measuring T0 once and using it for both D = 10 and D = 20...")
    shared_t0 = measure_T0(repeats=5)
    print(f">>> Shared T0 = {shared_t0:.6f} s")
    results = []

    for D in [10, 20]:
        res = run_complexity_test(D, t0=shared_t0)
        results.append(res)

    print_complexity_table(results)

    try:
        import pandas as pd

        rows = []
        for res in results:
            rows.append({
                "D": res["D"],
                "T0": res["T0"],
                "T1": res["T1"],
                "T2_mean": res["T2"],
                "Complexity": res["Complexity"],
                "T2_run_1": res["T2_runs"][0],
                "T2_run_2": res["T2_runs"][1],
                "T2_run_3": res["T2_runs"][2],
                "T2_run_4": res["T2_runs"][3],
                "T2_run_5": res["T2_runs"][4],
            })

        df = pd.DataFrame(rows)
        output_path = RESULTS_DIR / "PDABC_CEC2022_Complexity.csv"
        df.to_csv(output_path, index=False)
        print("\nSaved:")
        print(f" - {output_path}")

    except Exception as e:
        print("\nCould not save CSV:", e)
