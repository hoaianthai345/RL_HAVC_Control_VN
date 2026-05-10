# RL HVAC Control for Vietnamese Climate Zones

Repository for the manuscript and reproducibility artifacts for:

**Edge--Cloud Reinforcement Learning for Transferable HVAC Control Across Vietnamese Climate Zones**

Repository URL: <https://github.com/hoaianthai345/RL_HAVC_Control_VN>

## What is included

```text
manuscript_eaai/          LaTeX source, references, figures, tables, and compiled PDF
experiments_cloud/        Experiment code, configs, summary results, and policy artifacts
docs/                     Run notes and project documentation
outputs/                  Research/audit notes generated during preparation
notes/                    Figure/style notes
papers/                   Earlier paper-style drafts and reports
```

The canonical submission files are:

- `manuscript_eaai/main.tex`
- `manuscript_eaai/references.bib`
- `manuscript_eaai/main.bbl`
- `manuscript_eaai/main.pdf`
- `manuscript_eaai/figures/*.pdf`
- `manuscript_eaai/tables/*.tex`
- `manuscript_eaai/highlights.txt`
- `manuscript_eaai/cover_letter_draft.md`
- `manuscript_eaai/submission_checklist.md`

## Reproducibility notes

The manuscript uses HOT building archetypes and Vietnam weather files. Large local data, raw EnergyPlus outputs, virtual environments, and temporary logs are intentionally excluded by `.gitignore`.

Tracked experiment materials should include:

- `experiments_cloud/configs/` — context and run configuration files
- `experiments_cloud/src/` — experiment/metric code
- `experiments_cloud/scripts/` — plotting, table-generation, and run scripts
- `experiments_cloud/results/summary/` — aggregated CSV summaries used by the manuscript
- `experiments_cloud/artifacts/policies/` — trained policy artifacts retained for release

## Build the manuscript

From the repository root:

```bash
cd manuscript_eaai
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The compiled PDF is written to:

```text
manuscript_eaai/main.pdf
```

## Python environment

For experiment scripts:

```bash
cd experiments_cloud
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See `experiments_cloud/README_CLOUD_EXPERIMENTS.md` and `experiments_cloud/GPU_SERVER_RUNBOOK_VI.md` for run details.

## Recreate / audit summary tables and figures

The paper reports numbers from the aggregated CSV files in:

```text
experiments_cloud/results/summary/vietnam_final/
```

The most important manuscript source table is:

```text
experiments_cloud/results/summary/vietnam_final/control_summary.csv
```

Use the tracked summary CSVs to audit the reported manuscript numbers. If raw rollout CSVs are available locally, regenerate LaTeX table fragments and figures with:

```bash
cd experiments_cloud
python scripts/latex_tables.py \
  --input-dir results/raw/run_20260509_152412 \
  --output-dir ../manuscript_eaai/tables

python scripts/plot_results.py \
  --input-dir results/raw/run_20260509_152412 \
  --output-dir ../manuscript_eaai/figures \
  --plots all
```

Raw EnergyPlus outputs can be large and are ignored by default; the release keeps the compact summary CSVs needed to check the paper's reported values.

## Submission status

The repository is arranged so that `manuscript_eaai/` is the single canonical manuscript folder. Generated LaTeX auxiliaries and duplicate submission bundles are ignored; the final PDF and source files are kept.
