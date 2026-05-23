# Review Supplement Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the smallest high-value experiment and analysis package needed to answer likely reviewer objections about baselines, statistics, cost, and noise robustness.

**Architecture:** Keep the original training code unchanged where possible. Add a review supplement runner for new MLP-large and noisy-label experiments, add a review supplement builder that aggregates existing and new evidence into CSV/Markdown/LaTeX-ready text, then update the CIKM manuscript with the resulting numbers and honest boundaries.

**Tech Stack:** Python 3.10/3.13-compatible scripts, existing PyTorch training utilities in `experiments/run_tabular.py`, pandas/numpy/scipy when available, standard-library `unittest`, ACM LaTeX manuscript in `paper_cikm2026/main.tex`.

---

### Task 1: Add Tests For Statistical Helpers

**Files:**
- Create: `tests/test_review_supplement.py`
- Create: `scripts/build_review_supplement.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_review_supplement.py` with tests for geometric means, paired sign-test p-values, win counts, and finite mean formatting.

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\App1\environment\envs\ai_base\python.exe -m unittest tests.test_review_supplement -v`

Expected: import failure because `scripts.build_review_supplement` does not exist.

- [ ] **Step 3: Implement minimal helper functions**

Create `scripts/build_review_supplement.py` with helper functions only: `finite`, `mean_finite`, `geomean_finite`, `sign_test_p_value`, `paired_summary`, and formatting helpers.

- [ ] **Step 4: Run tests and verify pass**

Run: `D:\App1\environment\envs\ai_base\python.exe -m unittest tests.test_review_supplement -v`

Expected: all tests pass.

### Task 2: Build Existing-Evidence Supplement

**Files:**
- Modify: `scripts/build_review_supplement.py`
- Read: `results_v2/feynman_lowdim_clean12_pairwise_by_seed.csv`
- Read: `results_v2/synthetic_final/synthetic_final_pairwise_by_seed.csv`
- Read: `results_v2/srsd_grouped/srsd_full_pairwise_by_seed.csv`
- Read: `results_v2/fairness_audit_summary_20260509.csv`

- [ ] **Step 1: Add aggregation code**

Read existing pairwise and fairness files. Produce `results_v2/review_supplement/review_pairwise_significance.csv`, `review_cost_summary.csv`, and `review_supplement_report.md`.

- [ ] **Step 2: Run builder**

Run: `D:\App1\environment\envs\ai_base\python.exe scripts\build_review_supplement.py`

Expected: output directory and report exist; report lists Feynman, SRSD, synthetic test, synthetic OOD, and cost evidence.

### Task 3: Add Review Supplement Experiment Runner

**Files:**
- Create: `scripts/run_review_supplement.py`

- [ ] **Step 1: Implement MLP-large experiment**

Train `mlp_large` as an MLP model with larger width/depth on the equation-native suite using existing `train_one`. Save JSON/checkpoint files under `results_v2/review_supplement/mlp_large/equation_native`.

- [ ] **Step 2: Implement noise robustness experiment**

For selected equation-native datasets and seeds 0--2, train MLP, RBF-KAN, StableEML, and EML-KAN after adding deterministic Gaussian noise to the training labels only. Save JSON/checkpoint files under `results_v2/review_supplement/noise_robustness`.

- [ ] **Step 3: Ensure resume behavior**

Skip a run if a matching result already exists, so interrupted experiments can resume safely.

### Task 4: Run New Experiments

**Files Produced:**
- `results_v2/review_supplement/mlp_large/equation_native/*.json`
- `results_v2/review_supplement/noise_robustness/*.json`

- [ ] **Step 1: Run MLP-large**

Run a parameter-matched but stronger MLP on all ten equation-native functions, seeds 0--2.

- [ ] **Step 2: Run noise robustness**

Run a compact noise panel on five equation-native functions, seeds 0--2, noise levels 1%, 5%, and 10%, methods MLP/RBF-KAN/StableEML/EML-KAN.

### Task 5: Aggregate New Experiments

**Files:**
- Modify: `scripts/build_review_supplement.py`
- Read: `results_v2/review_supplement/mlp_large/equation_native/*.json`
- Read: `results_v2/review_supplement/noise_robustness/*.json`

- [ ] **Step 1: Add MLP-large and noise summary aggregators**

Produce `review_mlp_large_summary.csv` and `review_noise_robustness_summary.csv`.

- [ ] **Step 2: Re-run builder**

Run: `D:\App1\environment\envs\ai_base\python.exe scripts\build_review_supplement.py`

Expected: report includes the new MLP-large and noise sections.

### Task 6: Update Manuscript

**Files:**
- Modify: `paper_cikm2026/main.tex`
- Read: `results_v2/review_supplement/review_supplement_report.md`

- [ ] **Step 1: Add a compact review-supplement paragraph/table**

Insert statistical significance, cost, MLP-large, and noise robustness evidence without overclaiming.

- [ ] **Step 2: Tighten limitations**

Keep official pykan/KAN 2.0, AI-Feynman, long-budget symbolic search, and human audit usefulness framed as future/limitation unless directly supported.

### Task 7: Verify

**Commands:**
- `D:\App1\environment\envs\ai_base\python.exe -m unittest tests.test_review_supplement -v`
- `D:\App1\environment\envs\ai_base\python.exe scripts\build_review_supplement.py`
- LaTeX build command from `paper_cikm2026/README_CIKM2026.md`
- Log scan for fatal/undefined/overfull issues

**Completion Criteria:** Tests pass, report regenerates, manuscript builds, and all new numerical claims in `main.tex` are traceable to CSV/Markdown artifacts.
