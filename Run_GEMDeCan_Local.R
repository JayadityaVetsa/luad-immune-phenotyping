# ==============================================================================
# GEMDeCan Local Pipeline (Windows/Local Optimized) - RESTORED
# ==============================================================================

# 1. Setup & Licensing
# ------------------------------------------------------------------------------
print("[1/6] Initializing Environment...")

# SETUP LOCAL LIBRARY (Fixes permission issues)
local_lib <- file.path(getwd(), "R_libs")
if (!dir.exists(local_lib)) dir.create(local_lib)
.libPaths(c(local_lib, .libPaths()))
print(paste("Running with local library:", local_lib))

# Function to safely install if missing
ensure_package <- function(pkg, version=NULL, bioc=FALSE, git=NULL, archive_url=NULL) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    print(paste("Installing:", pkg))
    if (!is.null(archive_url)) {
      if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes")
      tryCatch(remotes::install_url(archive_url, upgrade="never"), error=function(e) print(e))
    } else if (!is.null(git)) {
      if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes")
      remotes::install_github(git, upgrade="never")
    } else if (bioc) {
      if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
      BiocManager::install(pkg, update=FALSE, ask=FALSE)
    } else {
      install.packages(pkg, repos="http://cran.rstudio.com/")
    }
  }
}

# Core Utilities
ensure_package("readr"); ensure_package("dplyr"); ensure_package("tidyr")
ensure_package("stringr"); ensure_package("magrittr"); ensure_package("purrr")
ensure_package("tibble"); ensure_package("remotes"); ensure_package("devtools")

# BioConductor & Deconvolution Tools
ensure_package("limSolve"); ensure_package("quadprog"); ensure_package("lpSolve")
ensure_package("EpiDISH", bioc=TRUE)
ensure_package("preprocessCore", bioc=TRUE)
ensure_package("GSVA", bioc=TRUE)

# DeconRNASeq (Removed from Bioc 3.20+) - Install from Archive
if (!requireNamespace("DeconRNASeq", quietly=TRUE)) {
  print("Installing DeconRNASeq from Bioconductor 3.18 Archive...")
  ensure_package("DeconRNASeq", archive_url="https://bioconductor.org/packages/3.18/bioc/src/contrib/DeconRNASeq_1.44.0.tar.gz")
}

# immunedeconv (from omnideconv)
ensure_package("immunedeconv", git="omnideconv/immunedeconv")

library(readr); library(dplyr); library(tibble); library(purrr); library(tidyr)
library(stringr); library(magrittr)

# 2. Clone GEMDeCan Repo
# ------------------------------------------------------------------------------
repo_dir <- "GEMDeCan_deconvolution"
if (dir.exists(file.path(repo_dir, ".git"))) {
  print("[2/6] GEMDeCan repository already exists. Skipping clone.")
} else {
  if (dir.exists(repo_dir)) unlink(repo_dir, recursive = TRUE)
  print("[2/6] Cloning GEMDeCan repository (requires Git in PATH)...")
  ret <- system("git clone https://github.com/VeraPancaldiLab/GEMDeCan_deconvolution.git")
  if (ret != 0) stop("Git clone failed! Do you have Git installed and in your PATH?")
}

# 3. ROBUST PATCHING (Crucial for Local R versions)
# ------------------------------------------------------------------------------
print("[3/6] Patching repository scripts for local compatibility...")
algo_script <- file.path(repo_dir, "scripts/deconvolution/deconvolution_algorithms.R")
lines <- readLines(algo_script)

# Remove any lines that force installation or set repos (conflicts with local factory)
lines <- lines[!grepl("BiocManager::install", lines)]
lines <- lines[!grepl("install.packages", lines)]
lines <- lines[!grepl("remotes::install", lines)]
lines <- lines[!grepl("devtools::install", lines)]
lines <- lines[!grepl("options\\(repos", lines)]

# Force explicit loading of stringr and others INSIDE the sourced script
lines <- c("library(stringr); library(dplyr); library(magrittr); library(tibble); library(tidyr)", lines)

writeLines(lines, algo_script)
print(" -> Script patched.")

# 4. Setup Signatures
# ------------------------------------------------------------------------------
print("[4/6] Setting up Signatures...")
sig_dir <- file.path(repo_dir, "scripts/deconvolution/signatures")
if (!dir.exists(sig_dir)) dir.create(sig_dir, recursive = TRUE)
lm22_file <- file.path(sig_dir, "LM22.txt")
if (!file.exists(lm22_file)) {
  download.file("https://raw.githubusercontent.com/mdozmorov/Immuno_notes/master/data/Cibersoft/LM22.txt", lm22_file)
}

# 5. Process Input Data
# ------------------------------------------------------------------------------
print("[5/6] Reading input .txt/.tsv files...")
all_files <- list.files(pattern = "\\.(txt|tsv|csv)$")
ignore_list <- c("deconvolution_results.csv", "LM22.txt", "deconvolution_results.txt", "input_matrix_merged.tsv", "all_samples_GEMDeCan_results.csv")
input_files <- all_files[!all_files %in% ignore_list]

if (length(input_files) == 0) {
  print("WARNING: No input files found to process yet.")
} else {
  read_clean <- function(f) {
    # Robust read
    d <- tryCatch(read_tsv(f, comment = "#", show_col_types = FALSE), error=function(e) read_csv(f, show_col_types=FALSE))
    
    # Check GDC vs Generic
    cols <- colnames(d)
    if ("gene_name" %in% cols) {
      d <- d %>% select(Gene = gene_name, Value = tpm_unstranded)
    } else {
      # Assume Col 1 = Gene, Col 2 = Value
      d <- d %>% select(Gene = 1, Value = 2)
    }
    
    # Average duplicates
    d %>%
      filter(!is.na(Gene)) %>%
      mutate(Value = as.numeric(Value)) %>%
      group_by(Gene) %>%
      summarise(Value = mean(Value, na.rm=TRUE)) %>%
      rename(!!tools::file_path_sans_ext(basename(f)) := Value)
  }

  print(paste(" -> Merging", length(input_files), "files..."))
  list_dfs <- lapply(input_files, read_clean)
  
  # Full Outer Join of all samples (Gene union)
  final_mx <- list_dfs %>% reduce(full_join, by = "Gene") %>%
    mutate(across(where(is.numeric), ~replace_na(., 0))) %>%
    column_to_rownames("Gene") %>%
    as.matrix()

  print(paste(" -> Matrix:", nrow(final_mx), "Genes x", ncol(final_mx), "Samples"))

  # 6. Run GEMDeCan Logic
  # ------------------------------------------------------------------------------
  print("[6/6] Executing Deconvolution Pipeline...")
  current_wd <- getwd()
  setwd(repo_dir) # Enter repo for relative paths
  
  tryCatch({
    source("scripts/deconvolution/deconvolution_algorithms.R")
  
  # Ensure libraries are attached AFTER sourcing
  library(immunedeconv)
  library(EpiDISH)
  library(DeconRNASeq)

  # Run Algorithms
    # A. Quantiseq
    print(" -> Running Quantiseq...")
    res_q <- tryCatch({
       computeQuantiseq(final_mx)
    }, error = function(e) {
       print(paste("WARNING: Quantiseq failed (skipping):", e$message))
       tibble(sample = colnames(final_mx))
    })
    
    # B. EpiDISH (Variable Signatures)
    print(" -> Running EpiDISH/DeconRNASeq...")
    sig_files <- list.files("scripts/deconvolution/signatures", full.names = TRUE)
    # Filter out directories (like MCPcounter/EPIC folders that get installed there)
    sig_files <- sig_files[!dir.exists(sig_files)]
    
    res_v <- methods_with_variable_signatures(final_mx, sig_files)
    
    # C. Merge
    final_df <- inner_join(res_q, res_v, by = "sample")
    
  }, finally = {
    setwd(current_wd) # Restore WD
  })
  
  write_csv(final_df, "all_samples_GEMDeCan_results.csv")
  print("=======================================================")
  print("SUCCESS! Results saved to: all_samples_GEMDeCan_results.csv")
  print("=======================================================")
  print(head(final_df))
}
