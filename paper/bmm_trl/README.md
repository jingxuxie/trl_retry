# BMM-TRL manuscript scaffold

This directory contains a lightweight LaTeX manuscript scaffold for the current
BMM-TRL evidence package.

Primary source files:

```text
main.tex
references.bib
../../assets/bmm_tabular_error_scaling.png
```

Current source-of-truth result artifacts:

```text
../../exp/bmm_paper_tables_final.md
../../exp/bmm_paper_tables_final.json
../../exp/bmm_advanced_policy_table.md
../../exp/bmm_advanced_policy_table.json
../../BMM_TRL_CONFERENCE_DRAFT.md
../../BMM_TRL_PAPER_CLAIM_PACKAGE.md
```

Validate and refresh the paper headline tables from the repo root:

```text
python scripts/validate_bmm_paper_claims.py
```

Preferred build command, if a local LaTeX toolchain is available:

```text
latexmk -pdf -interaction=nonstopmode main.tex
```

Manual fallback:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The manuscript intentionally frames policy results as fixed-controller
hierarchical planning evaluations and separates those results from end-to-end
actor extraction.
