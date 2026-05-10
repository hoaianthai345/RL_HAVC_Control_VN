# ICSCCT 2026 LaTeX Draft

This folder contains an English LaTeX draft for the HVAC edge--cloud RL paper.

## Files

- `main.tex` — compile-ready draft with results and visualization left as TODO placeholders.
- `references.bib` — BibTeX references and ICSCCT source pages.
- `figures/` — placeholder directory for generated figures.
- `tables/` — placeholder directory for exported tables if needed.

## Compile

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Submission note

ICSCCT 2026 indicates Springer formatting / LNNS proceedings. This draft uses a portable `article` class so it compiles locally. Before submission, transfer the content into the official Springer template required by ICSCCT.

## Current status

No experimental results or final visualizations are included. Fill those only after validated HOT/EnergyPlus experiments and latency measurements are complete.
