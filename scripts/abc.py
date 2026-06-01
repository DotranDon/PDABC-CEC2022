import numpy as np
import os
import argparse
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
import opfunu


# ============================================================
# RUN ABC ON CEC2022
#
# Output:
#   ABC_1_10.txt, ..., ABC_12_10.txt
#
# Each file: 17 x 30 matrix
#   rows 0..15 : function error F(x) - F*
#   row 16     : FEterm
# python abc.py --D 10  --workers 8
# ============================================================

EPS = 1e-8
RUNS = 1
NUM_FUNCTIONS = 12
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# ===================== 1. CONFIGURATION AND REFERENCE DATA =====================
# Theoretical optimum values (bias)
CEC2022_BIAS = {
    1: 300, 2: 400, 3: 600, 4: 800, 5: 900, 6: 1800,
    7: 2000, 8: 2200, 9: 2300, 10: 2400, 11: 2600, 12: 2700
}

# Load 1000 random seeds from Rand Seeds.txt file.
RAND_SEEDS = np.array([
        958.0, 128.0, 512.0, 166.0, 538.0, 894.0, 449.0, 195.0, 88.0, 144.0,
        903.0, 577.0, 830.0, 827.0, 537.0, 179.0, 660.0, 844.0, 999.0, 858.0,
        744.0, 740.0, 221.0, 443.0, 219.0, 484.0, 5.0, 23.0, 465.0, 864.0,
        245.0, 523.0, 804.0, 588.0, 922.0, 11.0, 746.0, 789.0, 558.0, 967.0,
        450.0, 205.0, 275.0, 376.0, 331.0, 389.0, 933.0, 496.0, 561.0, 751.0,
        167.0, 775.0, 460.0, 608.0, 250.0, 947.0, 541.0, 121.0, 932.0, 165.0,
        152.0, 801.0, 612.0, 696.0, 738.0, 862.0, 691.0, 257.0, 243.0, 836.0,
        283.0, 206.0, 51.0, 713.0, 526.0, 371.0, 102.0, 520.0, 881.0, 503.0,
        431.0, 594.0, 812.0, 365.0, 362.0, 439.0, 787.0, 511.0, 37.0, 605.0,
        416.0, 575.0, 811.0, 372.0, 185.0, 112.0, 748.0, 104.0, 314.0, 831.0,
        613.0, 168.0, 485.0, 384.0, 317.0, 184.0, 661.0, 60.0, 500.0, 292.0,
        46.0, 970.0, 700.0, 925.0, 415.0, 499.0, 695.0, 183.0, 961.0, 576.0,
        890.0, 544.0, 555.0, 194.0, 96.0, 587.0, 17.0, 758.0, 409.0, 624.0,
        217.0, 65.0, 30.0, 284.0, 973.0, 942.0, 182.0, 616.0, 883.0, 150.0,
        808.0, 424.0, 223.0, 78.0, 819.0, 633.0, 238.0, 228.0, 733.0, 137.0,
        174.0, 825.0, 763.0, 682.0, 976.0, 404.0, 673.0, 377.0, 960.0, 175.0,
        727.0, 997.0, 719.0, 148.0, 277.0, 795.0, 912.0, 201.0, 514.0, 214.0,
        615.0, 74.0, 865.0, 176.0, 209.0, 571.0, 645.0, 777.0, 728.0, 686.0,
        347.0, 293.0, 667.0, 923.0, 339.0, 584.0, 265.0, 532.0, 957.0, 968.0,
        455.0, 884.0, 146.0, 855.0, 312.0, 6.0, 1.0, 800.0, 648.0, 530.0,
        423.0, 89.0, 908.0, 467.0, 315.0, 669.0, 785.0, 773.0, 80.0, 525.0,
        846.0, 412.0, 762.0, 55.0, 988.0, 436.0, 764.0, 27.0, 677.0, 944.0,
        629.0, 792.0, 241.0, 684.0, 322.0, 421.0, 529.0, 735.0, 249.0, 459.0,
        729.0, 191.0, 356.0, 507.0, 278.0, 225.0, 610.0, 926.0, 835.0, 517.0,
        48.0, 985.0, 398.0, 948.0, 261.0, 705.0, 311.0, 461.0, 747.0, 288.0,
        806.0, 93.0, 341.0, 57.0, 521.0, 843.0, 850.0, 915.0, 28.0, 494.0,
        401.0, 510.0, 631.0, 992.0, 782.0, 856.0, 33.0, 900.0, 809.0, 954.0,
        859.0, 918.0, 405.0, 16.0, 471.0, 36.0, 346.0, 452.0, 676.0, 994.0,
        3.0, 518.0, 750.0, 136.0, 417.0, 489.0, 734.0, 504.0, 342.0, 438.0,
        67.0, 52.0, 982.0, 229.0, 361.0, 524.0, 768.0, 603.0, 263.0, 454.0,
        351.0, 81.0, 863.0, 334.0, 10.0, 598.0, 75.0, 109.0, 84.0, 445.0,
        232.0, 396.0, 851.0, 437.0, 557.0, 841.0, 94.0, 192.0, 380.0, 25.0,
        871.0, 58.0, 466.0, 85.0, 671.0, 15.0, 820.0, 852.0, 642.0, 492.0,
        771.0, 559.0, 878.0, 767.0, 470.0, 379.0, 129.0, 741.0, 276.0, 329.0,
        242.0, 501.0, 432.0, 338.0, 876.0, 591.0, 815.0, 680.0, 534.0, 759.0,
        280.0, 270.0, 788.0, 316.0, 453.0, 110.0, 325.0, 378.0, 186.0, 327.0,
        260.0, 56.0, 216.0, 64.0, 106.0, 590.0, 564.0, 13.0, 478.0, 352.0,
        178.0, 24.0, 43.0, 659.0, 69.0, 299.0, 363.0, 628.0, 120.0, 38.0,
        885.0, 483.0, 893.0, 891.0, 692.0, 26.0, 873.0, 566.0, 406.0, 29.0,
        635.0, 962.0, 621.0, 188.0, 791.0, 874.0, 638.0, 593.0, 823.0, 861.0,
        45.0, 539.0, 53.0, 547.0, 928.0, 653.0, 248.0, 807.0, 679.0, 8.0,
        224.0, 895.0, 422.0, 282.0, 582.0, 297.0, 97.0, 50.0, 567.0, 273.0,
        704.0, 508.0, 637.0, 910.0, 427.0, 419.0, 896.0, 123.0, 614.0, 837.0,
        662.0, 761.0, 289.0, 111.0, 231.0, 887.0, 211.0, 154.0, 197.0, 930.0,
        375.0, 66.0, 611.0, 984.0, 562.0, 701.0, 193.0, 879.0, 87.0, 113.0,
        482.0, 196.0, 602.0, 117.0, 83.0, 189.0, 754.0, 606.0, 303.0, 295.0,
        528.0, 977.0, 509.0, 585.0, 952.0, 688.0, 814.0, 834.0, 386.0, 147.0,
        817.0, 737.0, 678.0, 569.0, 458.0, 986.0, 640.0, 21.0, 664.0, 240.0,
        950.0, 929.0, 163.0, 272.0, 493.0, 716.0, 259.0, 583.0, 253.0, 441.0,
        725.0, 187.0, 515.0, 222.0, 897.0, 755.0, 798.0, 410.0, 907.0, 358.0,
        549.0, 480.0, 832.0, 902.0, 578.0, 207.0, 715.0, 860.0, 181.0, 527.0,
        162.0, 435.0, 996.0, 506.0, 252.0, 572.0, 100.0, 139.0, 279.0, 132.0,
        333.0, 392.0, 813.0, 236.0, 772.0, 974.0, 531.0, 833.0, 244.0, 953.0,
        210.0, 690.0, 235.0, 420.0, 324.0, 955.0, 321.0, 164.0, 920.0, 18.0,
        636.0, 888.0, 9.0, 490.0, 935.0, 335.0, 989.0, 349.0, 938.0, 70.0,
        870.0, 545.0, 711.0, 247.0, 430.0, 426.0, 708.0, 105.0, 300.0, 287.0,
        456.0, 4.0, 336.0, 689.0, 868.0, 103.0, 305.0, 481.0, 374.0, 654.0,
        239.0, 548.0, 625.0, 425.0, 290.0, 157.0, 753.0, 707.0, 320.0, 875.0,
        730.0, 916.0, 580.0, 599.0, 497.0, 254.0, 672.0, 749.0, 170.0, 828.0,
        42.0, 905.0, 592.0, 203.0, 757.0, 770.0, 736.0, 717.0, 54.0, 687.0,
        306.0, 49.0, 570.0, 464.0, 627.0, 766.0, 649.0, 399.0, 936.0, 971.0,
        418.0, 227.0, 161.0, 302.0, 230.0, 816.0, 90.0, 993.0, 889.0, 769.0,
        469.0, 760.0, 712.0, 551.0, 350.0, 411.0, 190.0, 502.0, 323.0, 596.0,
        296.0, 706.0, 337.0, 359.0, 118.0, 212.0, 12.0, 171.0, 301.0, 619.0,
        59.0, 255.0, 366.0, 724.0, 987.0, 927.0, 62.0, 414.0, 702.0, 99.0,
        119.0, 142.0, 840.0, 126.0, 963.0, 226.0, 838.0, 19.0, 138.0, 319.0,
        793.0, 204.0, 477.0, 86.0, 79.0, 442.0, 39.0, 82.0, 143.0, 198.0,
        313.0, 924.0, 655.0, 778.0, 472.0, 382.0, 173.0, 457.0, 632.0, 330.0,
        271.0, 107.0, 574.0, 218.0, 267.0, 360.0, 355.0, 869.0, 158.0, 160.0,
        407.0, 934.0, 821.0, 63.0, 783.0, 670.0, 786.0, 546.0, 822.0, 140.0,
        448.0, 145.0, 141.0, 291.0, 381.0, 983.0, 473.0, 369.0, 917.0, 400.0,
        486.0, 685.0, 620.0, 124.0, 513.0, 269.0, 586.0, 32.0, 47.0, 980.0,
        643.0, 149.0, 446.0, 805.0, 756.0, 542.0, 281.0, 784.0, 810.0, 732.0,
        2.0, 966.0, 429.0, 159.0, 799.0, 623.0, 135.0, 939.0, 373.0, 824.0,
        7.0, 383.0, 681.0, 951.0, 595.0, 462.0, 274.0, 626.0, 726.0, 709.0,
        573.0, 668.0, 941.0, 714.0, 91.0, 368.0, 710.0, 487.0, 522.0, 151.0,
        476.0, 990.0, 535.0, 601.0, 723.0, 213.0, 652.0, 344.0, 76.0, 802.0,
        474.0, 428.0, 125.0, 658.0, 394.0, 114.0, 345.0, 665.0, 153.0, 866.0,
        956.0, 683.0, 563.0, 552.0, 892.0, 979.0, 479.0, 847.0, 286.0, 780.0,
        550.0, 246.0, 116.0, 77.0, 332.0, 998.0, 101.0, 41.0, 752.0, 395.0,
        251.0, 488.0, 657.0, 304.0, 882.0, 68.0, 519.0, 940.0, 568.0, 61.0,
        898.0, 553.0, 491.0, 666.0, 991.0, 886.0, 127.0, 298.0, 849.0, 639.0,
        1000.0, 233.0, 589.0, 264.0, 387.0, 307.0, 385.0, 597.0, 617.0, 451.0,
        919.0, 367.0, 262.0, 650.0, 115.0, 310.0, 857.0, 172.0, 694.0, 402.0,
        234.0, 790.0, 22.0, 794.0, 641.0, 388.0, 872.0, 208.0, 880.0, 853.0,
        468.0, 309.0, 133.0, 981.0, 391.0, 237.0, 498.0, 408.0, 722.0, 779.0,
        803.0, 931.0, 256.0, 978.0, 199.0, 867.0, 268.0, 842.0, 533.0, 651.0,
        579.0, 495.0, 20.0, 911.0, 340.0, 845.0, 969.0, 200.0, 397.0, 354.0,
        909.0, 403.0, 348.0, 390.0, 731.0, 258.0, 818.0, 393.0, 720.0, 540.0,
        913.0, 357.0, 781.0, 475.0, 444.0, 155.0, 92.0, 328.0, 600.0, 353.0,
        72.0, 308.0, 630.0, 854.0, 622.0, 581.0, 180.0, 921.0, 796.0, 699.0,
        40.0, 674.0, 604.0, 937.0, 972.0, 906.0, 543.0, 505.0, 413.0, 98.0,
        434.0, 554.0, 698.0, 556.0, 73.0, 156.0, 826.0, 675.0, 647.0, 433.0,
        739.0, 560.0, 44.0, 440.0, 364.0, 318.0, 326.0, 965.0, 914.0, 95.0,
        959.0, 644.0, 718.0, 743.0, 131.0, 839.0, 693.0, 946.0, 634.0, 899.0,
        536.0, 904.0, 463.0, 266.0, 565.0, 285.0, 945.0, 34.0, 697.0, 656.0,
        663.0, 370.0, 995.0, 215.0, 122.0, 169.0, 177.0, 774.0, 646.0, 877.0,
        975.0, 742.0, 745.0, 721.0, 964.0, 220.0, 797.0, 130.0, 776.0, 618.0,
        35.0, 901.0, 703.0, 447.0, 71.0, 609.0, 343.0, 14.0, 516.0, 202.0,
        829.0, 848.0, 943.0, 134.0, 294.0, 108.0, 949.0, 607.0, 765.0, 31.0
])


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ABCConfig:
    D: int
    SN: int
    FE_max: int
    limit: int
    milestones: np.ndarray
    actual_seed: int


# ============================================================
# BASIC UTILITIES
# ============================================================

def ensure_bounds(x, lo=-100.0, hi=100.0):
    """
    Reflection + clipping boundary handling.
    """
    x_new = x.copy()

    over_hi = x_new > hi
    x_new[over_hi] = 2.0 * hi - x_new[over_hi]

    under_lo = x_new < lo
    x_new[under_lo] = 2.0 * lo - x_new[under_lo]

    return np.clip(x_new, lo, hi)


def fit_transform(fvals):
    """
    ABC fitness transform for minimization.
    """
    fvals = np.asarray(fvals, dtype=float)
    fvals = np.nan_to_num(fvals, nan=1e300, posinf=1e300, neginf=0.0)

    return np.where(
        fvals >= 0.0,
        1.0 / (1.0 + fvals),
        1.0 + np.abs(fvals)
    )


def init_pop(SN, D, rng):
    return -100.0 + 200.0 * rng.random((SN, D))


def record_progress(current_fe, current_f_best, milestones, errors_list, m_idx):
    """
    Record function error at CEC2022 milestones.
    """
    while m_idx < 16 and current_fe >= milestones[m_idx]:
        if current_f_best <= EPS:
            val = EPS
        else:
            val = max(float(current_f_best), 0.0)

        errors_list.append(val)
        m_idx += 1

    return m_idx


def cec2022_milestones(D, FE_max):
    """
    16 CEC2022 recording points:
        floor(D^(k/5 - 3) * MaxFES), k = 0..15
    """
    return np.floor([
        (D ** (k / 5.0 - 3.0)) * FE_max
        for k in range(16)
    ]).astype(int)


def get_cec_seed(D, func_no, run_id):
    """
    CEC2022 seed rule:

    MATLAB logic:
        seed_temp = (problem_size/10 * func_no * Runs + run_id) - Runs
        seed_ind  = mod(seed_temp, 1000) + 1
        run_seed  = Rand_Seeds(seed_ind)

    Python:
        seed_ind is 1-based, so use seed_ind - 1.
    """
    if len(RAND_SEEDS) != 1000:
        raise ValueError(
            f"RAND_SEEDS phải có đúng 1000 giá trị. "
            f"Hiện tại có {len(RAND_SEEDS)} giá trị."
        )

    seed_temp = int((D / 10.0 * func_no * RUNS + run_id) - RUNS)
    seed_ind = (seed_temp % 1000) + 1

    return int(RAND_SEEDS[seed_ind - 1])


# ============================================================
# ABC SEARCH EQUATION
# ============================================================

def generate_abc_candidate(X, i, rng):
    """
    ABC standard equation:
        v_j = x_ij + phi * (x_ij - x_kj)

    where:
        phi ~ U[-1, 1]
    """
    SN, D = X.shape

    k = rng.integers(0, SN)
    while k == i:
        k = rng.integers(0, SN)

    j = rng.integers(0, D)
    phi = rng.uniform(-1.0, 1.0)

    v = X[i].copy()
    v[j] = X[i, j] + phi * (X[i, j] - X[k, j])

    return ensure_bounds(v)


# ============================================================
# ABC CORE
# ============================================================

def ABC_Core(func, cfg: ABCConfig):
    """
    Core for canonical ABC.
    """
    D, SN, FE_max = cfg.D, cfg.SN, cfg.FE_max
    rng = np.random.default_rng(cfg.actual_seed)

    # -----------------------------
    # Initialization
    # -----------------------------
    X = init_pop(SN, D, rng)
    f = np.array([func(xi) for xi in X], dtype=float)

    fe = SN
    trials = np.zeros(SN, dtype=int)

    best_idx = int(np.argmin(f))
    f_best = float(f[best_idx])
    x_best = X[best_idx].copy()

    errors = []
    m_idx = 0
    fe_term = FE_max

    m_idx = record_progress(fe, f_best, cfg.milestones, errors, m_idx)

    if f_best <= EPS:
        fe_term = fe

    # -----------------------------
    # Main loop
    # -----------------------------
    while fe < FE_max and f_best > EPS:

        # =====================================================
        # EMPLOYED BEE PHASE
        # =====================================================
        for i in range(SN):
            if fe >= FE_max or f_best <= EPS:
                break

            v = generate_abc_candidate(
                X=X,
                i=i,
                rng=rng,
            )

            fv = float(func(v))
            fe += 1

            if fv <= f[i]:
                X[i] = v
                f[i] = fv
                trials[i] = 0

                if fv < f_best:
                    f_best = fv
                    x_best = v.copy()
            else:
                trials[i] += 1

            m_idx = record_progress(fe, f_best, cfg.milestones, errors, m_idx)

            if f_best <= EPS:
                fe_term = fe
                break

        if fe >= FE_max or f_best <= EPS:
            break

        # =====================================================
        # ONLOOKER BEE PHASE
        # =====================================================
        fit = fit_transform(f)
        fit_sum = float(np.sum(fit))

        if fit_sum <= 0.0 or not np.isfinite(fit_sum):
            probs = np.ones(SN, dtype=float) / SN
        else:
            probs = fit / fit_sum

        # Standard ABC style: SN onlookers choose food sources by probability
        onlookers = rng.choice(SN, size=SN, replace=True, p=probs)

        for i in onlookers:
            if fe >= FE_max or f_best <= EPS:
                break

            v = generate_abc_candidate(
                X=X,
                i=int(i),
                rng=rng,
            )

            fv = float(func(v))
            fe += 1

            if fv <= f[i]:
                X[i] = v
                f[i] = fv
                trials[i] = 0

                if fv < f_best:
                    f_best = fv
                    x_best = v.copy()
            else:
                trials[i] += 1

            m_idx = record_progress(fe, f_best, cfg.milestones, errors, m_idx)

            if f_best <= EPS:
                fe_term = fe
                break

        if fe >= FE_max or f_best <= EPS:
            break

        # =====================================================
        # SCOUT BEE PHASE
        # =====================================================
        # Theo framework ABC: chọn một nghiệm inactive nhất để thay thế.
        worst_trial_idx = int(np.argmax(trials))

        if trials[worst_trial_idx] >= cfg.limit and fe < FE_max and f_best > EPS:
            i = worst_trial_idx

            X[i] = init_pop(1, D, rng)[0]
            f[i] = float(func(X[i]))
            fe += 1
            trials[i] = 0

            if f[i] < f_best:
                f_best = f[i]
                x_best = X[i].copy()

            m_idx = record_progress(fe, f_best, cfg.milestones, errors, m_idx)

            if f_best <= EPS:
                fe_term = fe
                break

    # -----------------------------
    # Fill remaining CEC records
    # -----------------------------
    if f_best <= EPS:
        fill_value = EPS
    else:
        fill_value = max(float(f_best), 0.0)

    while len(errors) < 16:
        errors.append(fill_value)

    return errors + [fe_term]


def ABC_standard_Core(func, cfg: ABCConfig):
    return ABC_Core(func, cfg)


# ============================================================
# ONE EXPERIMENT TASK
# ============================================================

def run_experiment(task):
    algo_name, fid, D, rid, MaxFES, milestones, SN, limit_factor = task

    actual_seed = get_cec_seed(D, fid, rid)

    problem_class = getattr(opfunu.cec_based.cec2022, f"F{fid}2022")
    problem = problem_class(ndim=D)

    f_star = CEC2022_BIAS[fid]

    def obj_func(x):
        err = float(problem.evaluate(x)) - f_star
        return max(err, 0.0)

    limit = int(limit_factor * SN * D)

    cfg = ABCConfig(
        D=D,
        SN=SN,
        FE_max=MaxFES,
        limit=limit,
        milestones=milestones,
        actual_seed=actual_seed,
    )

    if algo_name == "ABC":
        results = ABC_standard_Core(obj_func, cfg)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    return algo_name, fid, rid, results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run ABC on CEC2022 benchmark."
    )

    parser.add_argument(
        "--D",
        type=int,
        choices=[10, 20],
        required=True,
        help="Dimension: 10 or 20."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers."
    )

    parser.add_argument(
        "--SN",
        type=int,
        default=30,
        help="Number of food sources / population size."
    )

    parser.add_argument(
        "--limit_factor",
        type=float,
        default=0.5,
        help="limit = limit_factor * SN * D. Default follows your previous setup."
    )

    parser.add_argument(
        "--algos",
        nargs="+",
        choices=["ABC"],
        default=["ABC"],
        help="Algorithms to run."
    )

    args = parser.parse_args()

    D = args.D
    MaxFES = 200_000 if D == 10 else 1_000_000
    milestones = cec2022_milestones(D, MaxFES)

    print("=" * 90)
    print("CEC2022 ABC RUN")
    print("=" * 90)
    print(f"D             : {D}")
    print(f"MaxFES        : {MaxFES}")
    print(f"Runs          : {RUNS}")
    print(f"SN            : {args.SN}")
    print(f"Limit         : int({args.limit_factor} * SN * D) = {int(args.limit_factor * args.SN * D)}")
    print(f"Workers       : {args.workers}")
    print(f"Algorithms    : {args.algos}")
    print(f"Output dir    : {RESULTS_DIR.resolve()}")
    print("=" * 90)

    for algo_name in args.algos:
        print(f"\n>>> Running {algo_name} for D={D}...")

        final_data = {
            fid: np.zeros((17, RUNS), dtype=float)
            for fid in range(1, NUM_FUNCTIONS + 1)
        }

        tasks = [
            (
                algo_name,
                fid,
                D,
                rid,
                MaxFES,
                milestones,
                args.SN,
                args.limit_factor,
            )
            for fid in range(1, NUM_FUNCTIONS + 1)
            for rid in range(1, RUNS + 1)
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_experiment, task) for task in tasks]

            for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                algo, fid, rid, results = future.result()

                final_data[fid][:, rid - 1] = results

                print(
                    f"[{count:03d}/{len(tasks)}] "
                    f"{algo} | F{fid:02d} | Run {rid:02d} | "
                    f"Final Error: {results[15]:.6e} | FEterm: {results[16]:.0f}"
                )

        print(f"\n>>> Saving {algo_name} result files...")

        for fid in range(1, NUM_FUNCTIONS + 1):
            filename = RESULTS_DIR / f"{algo_name}_{fid}_{D}.txt"

            np.savetxt(
                filename,
                final_data[fid],
                fmt="%.8e",
                delimiter="\t"
            )

            print(f"Saved: {filename}")

    print("\nDone.")


if __name__ == "__main__":
    main()