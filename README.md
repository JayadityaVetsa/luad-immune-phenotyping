# LUAD immune phenotyping

Code and manuscript source for a retrospective analysis of immune phenotypes and overall survival in TCGA lung adenocarcinoma (TCGA-LUAD) bulk RNA-seq profiles.

The workflow integrates transcriptome deconvolution features, engineered immune-landscape metrics, phenotype clustering, supervised reproduction of phenotype assignments, survival analysis, and an exploratory immune-cell interaction network score. The reported associations are retrospective and internally evaluated; they are not a clinical decision tool.

## Repository layout

- `paper/`: self-contained LaTeX manuscript, bibliography, figures, and final PDF
- `results/`: processed result summaries and publication figures
- `pipeline_core.py`, `novel_*.py`, `survival_analysis.py`: principal analysis modules
- `requirements.txt`: Python dependencies
- `Run_GEMDeCan_*.R`: R entry points used for deconvolution

Raw TCGA data, local environments, installed R libraries, and trained model binaries are intentionally excluded. TCGA-LUAD source data are available from the [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/).

## Reproducibility

The repository preserves the code and processed outputs used for the manuscript. Re-running the complete workflow requires obtaining the source data described in the manuscript and installing both Python and R dependencies. No new experiments were run during preparation of this archival release.

The final manuscript is `paper/luad_immune_phenotyping_preprint.pdf`. Compile the source from `paper/` with `latexmk -pdf main.tex`.

Use the metadata in `CITATION.cff` when citing the software. A version DOI will be added after archival deposit.

Code is released under the MIT License. The manuscript text and figures are released under CC BY 4.0.
