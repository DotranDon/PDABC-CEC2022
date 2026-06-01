# PDABC-CEC2022

This repository provides source code and supporting files for the manuscript:

**“Probability-Dimension Artificial Bee Colony Algorithm for Engineering-Oriented Numerical Optimization”**

The repository is intended to support reproducibility and reviewer inspection. It contains the author-developed implementation of the proposed Probability-Dimension Artificial Bee Colony (PDABC) algorithm, the canonical ABC baseline, and supporting scripts used for score calculation, statistical testing, and computational complexity analysis.

## Overview

The proposed PDABC algorithm is a simple extension of the canonical Artificial Bee Colony (ABC) framework. It introduces:

* dimension-wise success memory;
* probability-guided dimension selection;
* a progress-dependent number of updated dimensions;
* modified employed, onlooker, and scout bee phases;
* lightweight vector-based updating without covariance matrix estimation or expensive local search.

The method is evaluated in the manuscript on the CEC2022 single-objective bound-constrained benchmark suite for dimensions `D=10` and `D=20`.

## Repository structure

The repository uses a simple script-based structure. The `scripts/` folder contains standalone Python scripts for running PDABC, running the ABC baseline, computing CEC2022-style target-reaching scores, performing statistical tests, and measuring computational complexity.

```text
PDABC-CEC2022/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── scripts/
│   ├── pdabc.py
│   ├── abc.py
│   ├── cec2022_target_reaching_score.py
│   ├── cec2022_statistical_tests.py
│   └── pdabc_complexity_test.py
│
├── results/
│   ├── PDABC_1_10.txt
│   ├── ...
│   ├── PDABC_12_20.txt
│   ├── ABC_1_10.txt
│   ├── ...
│   ├── ABC_12_20.txt
│   ├── CEC2022_Overall_Summary_Report.csv
│   ├── CEC2022_Function_Scores.csv
│   ├── CEC2022_Pairwise_Scores.csv
│   ├── CEC2022_All_Trial_Data.csv
│   └── PDABC_reduced2_CEC2022_Complexity_TR.csv
│
└── external/
    └── README.md
```

No separate `src/` package is required because the main algorithm scripts are self-contained and can be executed directly.

## External CEC2022 files

The official CEC2022 benchmark files and official competition result files are **not redistributed** in this repository.

Please obtain the official CEC2022 benchmark definitions, source files, and related materials from the official benchmark repository:

```text
https://github.com/P-N-Suganthan/2022-SO-BO
```

After downloading any required external competition result files, place the local copies in the `results/` folder only for running the score and statistical scripts. The PDABC and ABC scripts use the CEC2022 functions through `opfunu`; therefore, the official benchmark source files are not redistributed in this repository.

This repository only provides the author-developed PDABC/ABC implementation and supporting scripts. Any external CEC2022 files should be obtained from their original source.

## Requirements

Python 3.10 or later is recommended.

Install the required packages with:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` should include:

```text
numpy
pandas
scipy
matplotlib
opfunu
```

## Scripts

### 1. Run PDABC

The proposed Probability-Dimension Artificial Bee Colony algorithm is implemented in:

```text
scripts/pdabc.py
```

Example commands:

```bash
python scripts/pdabc.py --D 10 --workers 8 
python scripts/pdabc.py --D 20 --workers 8 
```
The value of --workers can be adjusted according to the number of available CPU cores. By default, the output files are written to the results/ folder.

The script generates output files such as:

```text
PDABC_1_10.txt, ..., PDABC_12_10.txt
PDABC_1_20.txt, ..., PDABC_12_20.txt
```

Each output file is a `17 x 30` matrix:

```text
rows 0..15 : function error values at CEC2022 recording points
row  16    : FEterm
```

### 2. Run the ABC baseline

The canonical ABC baseline is implemented in:

```text
scripts/abc.py
```

Example commands:

```bash
python scripts/abc.py --D 10 --workers 8 
python scripts/abc.py --D 20 --workers 8 
```
The value of --workers can be adjusted according to the number of available CPU cores. By default, the output files are written to the results/ folder.

The script generates output files such as:

```text
ABC_1_10.txt, ..., ABC_12_10.txt
ABC_1_20.txt, ..., ABC_12_20.txt
```

Each output file is a `17 x 30` matrix:

```text
rows 0..15 : function error values at CEC2022 recording points
row  16    : FEterm
```

### 3. Compute CEC2022 target-reaching scores

The CEC2022-style target-reaching score calculation is implemented in:

```text
scripts/cec2022_target_reaching_score.py
```
The script reads input result matrices from the results/ folder and writes the processed score files to the same folder.
This script compares algorithm results according to the CEC2022 target-reaching rule:

* if one trial reaches `EPS = 1e-8` and the other does not, the reached trial is better;
* if both trials reach `EPS`, the trial with the smaller `FEterm` is better;
* if neither trial reaches `EPS`, the trial with the smaller final error is better;
* exact ties give `0.5` point to each algorithm.

Example command:

```bash
python scripts/cec2022_target_reaching_score.py
```

The script expects result files in the form:

```text
AlgorithmName_FunctionNo_D.txt
```

For example:

```text
PDABC_1_10.txt
ABC_1_10.txt
EA4eigN100_10_1_10.txt
```

The script produces:

```text
CEC2022_Overall_Summary_Report.csv
CEC2022_Function_Scores.csv
CEC2022_Pairwise_Scores.csv
CEC2022_All_Trial_Data.csv
```

If external CEC2022 competition result files are needed for the full nine-algorithm comparison, they should be obtained from the official CEC2022 source and placed locally. They are not redistributed in this repository.

### 4. Statistical tests

Statistical analysis is implemented in:

```text
scripts/cec2022_statistical_tests.py
```

This script is used to compute Wilcoxon comparisons, Friedman ranks, and Holm post hoc analysis from the processed CEC2022 target-reaching trial data.

Example command:

```bash
python scripts/cec2022_statistical_tests.py
```

Typical output files may include:

```text
PDABC_TR_Wilcoxon_Summary.csv
PDABC_TR_Friedman_Ranks.csv
PDABC_TR_Holm_Posthoc.csv
```

The exact output file names may depend on the released script version.

### 5. Computational complexity test

The computational complexity measurement for PDABC is implemented in:

```text
scripts/pdabc_complexity_test.py
```

Example command:

```bash
python scripts/pdabc_complexity_test.py
```

This script measures the CEC-style computational complexity quantities, including `T0`, `T1`, repeated `T2` runs, mean `T2`, and the complexity indicator.

Typical output may include:

```text
PDABC_reduced2_CEC2022_Complexity_TR.csv
```

## Experimental settings

The main experimental settings used in the manuscript are:

| Setting                                 |                                                   Value |
| --------------------------------------- | ------------------------------------------------------: |
| Benchmark suite                         | CEC2022 single-objective bound-constrained optimization |
| Dimensions                              |                                          `D=10`, `D=20` |
| Number of benchmark functions           |                                                      12 |
| Independent runs                        |                                                      30 |
| Maximum function evaluations for `D=10` |                                                  200000 |
| Maximum function evaluations for `D=20` |                                                 1000000 |
| Number of food sources for PDABC/ABC    |                                                      30 |
| Scout limit                             |                                          `0.5 * SN * D` |

## Result files

The `results/` folder stores all input and output result files used by the scripts in this repository. No subfolder structure is required.

PDABC result files generated by `scripts/pdabc.py` include:

```text
PDABC_1_10.txt, ..., PDABC_12_10.txt
PDABC_1_20.txt, ..., PDABC_12_20.txt
```

ABC result files generated by `scripts/abc.py` include:

```text
ABC_1_10.txt, ..., ABC_12_10.txt
ABC_1_20.txt, ..., ABC_12_20.txt
```

Each PDABC or ABC result file is a `17 x 30` matrix:

```text
rows 0..15 : function error values at CEC2022 recording points
row  16    : FEterm
```

The target-reaching score script `scripts/cec2022_target_reaching_score.py` generates the following files in `results/`:

```text
CEC2022_Overall_Summary_Report.csv
CEC2022_Function_Scores.csv
CEC2022_Pairwise_Scores.csv
CEC2022_All_Trial_Data.csv
```

The statistical testing script `scripts/cec2022_statistical_tests.py` generates Wilcoxon, Friedman, and Holm post hoc result files in `results/`.

The complexity script `scripts/pdabc_complexity_test.py` generates the PDABC computational complexity summary in `results/`.

External CEC2022 competition result files required for reproducing the full nine-algorithm comparison are not redistributed in this repository. If needed, they should be obtained from the official CEC2022 source and placed locally in `results/` only for running the score and statistical scripts.

## Main manuscript result

In the manuscript, PDABC was compared with the canonical ABC baseline and seven algorithms selected from the CEC2022 competition. Under the CEC2022-style target-reaching score, PDABC obtained the highest overall score among the nine compared algorithms.

The main score summary reported in the manuscript is:

| Algorithm        |  `D=10` |  `D=20` | Total score |
| ---------------- | ------: | ------: | ----------: |
| PDABC            | 57613.5 | 58389.0 |    116002.5 |
| NL-SHADE-LBC     | 59693.5 | 49009.5 |    108703.0 |
| EA4eigN100_10    | 54354.0 | 54123.0 |    108477.0 |
| NL-SHADE-RSP-MID | 47423.0 | 40803.0 |     88226.0 |
| S_LSHADE_DP      | 40325.0 | 44967.0 |     85292.0 |
| ABC              | 36193.5 | 48649.0 |     84842.5 |
| NLSOMACLP        | 33002.0 | 32602.0 |     65604.0 |
| ZOCMAES          | 32727.5 | 32837.0 |     65564.5 |
| Co-PPSO          | 27468.0 | 27420.5 |     54888.5 |

These values are provided to help readers connect the repository with the manuscript tables.

## Notes on reproducibility

The full comparison in the manuscript uses:

1. author-generated PDABC results;
2. author-generated ABC baseline results;
3. selected official CEC2022 competition result files.

Only the author-generated code and results are distributed here. External benchmark files and official competition result files should be obtained from the official CEC2022 source.

## Citation

If you use this code or result files, please cite the associated manuscript:

```text
Don T. Do and Bao N. Do,
“Probability-Dimension Artificial Bee Colony Algorithm for Engineering-Oriented Numerical Optimization,”
submitted to Sādhanā – Academy Proceedings in Engineering Sciences.
```

A formal citation will be updated after publication.

## License

This repository is released for research and reproducibility purposes. See the `LICENSE` file for details.
