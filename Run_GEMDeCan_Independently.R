# ==============================================================================
# GEMDeCan Independent Pipeline (Per-Patient Execution)
# ==============================================================================

# 1. Setup & Licensing
# ------------------------------------------------------------------------------
print("[1/5] Initializing Environment...")

# SETUP LOCAL LIBRARY (Fixes permission issues)
local_lib <- file.path(getwd(), "R_libs")
if (!dir.exists(local_lib)) dir.create(local_lib)
.libPaths(c(local_lib, .libPaths()))
print(paste("Running with local library:", local_lib))

# --- Dependency Management ---
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

# Deconvolution Tools
ensure_package("limSolve"); ensure_package("quadprog"); ensure_package("lpSolve")
ensure_package("EpiDISH", bioc=TRUE)
ensure_package("preprocessCore", bioc=TRUE)
ensure_package("GSVA", bioc=TRUE)

if (!requireNamespace("DeconRNASeq", quietly=TRUE)) {
  print("Installing DeconRNASeq from Bioconductor 3.18 Archive...")
  ensure_package("DeconRNASeq", archive_url="https://bioconductor.org/packages/3.18/bioc/src/contrib/DeconRNASeq_1.44.0.tar.gz")
}
ensure_package("immunedeconv", git="omnideconv/immunedeconv")

library(readr); library(dplyr); library(tibble); library(purrr); library(tidyr)
library(stringr); library(magrittr)

# 2. Setup GEMDeCan Repository (Clone & Patch)
# ------------------------------------------------------------------------------
repo_dir <- "GEMDeCan_deconvolution"
if (dir.exists(file.path(repo_dir, ".git"))) {
  print("[2/5] GEMDeCan repository found. Skipping clone.")
} else {
  if (dir.exists(repo_dir)) unlink(repo_dir, recursive = TRUE)
  print("[2/5] Cloning GEMDeCan repository...")
  system("git clone https://github.com/VeraPancaldiLab/GEMDeCan_deconvolution.git")
}

print("[3/5] Patching scripts...")
algo_script <- file.path(repo_dir, "scripts/deconvolution/deconvolution_algorithms.R")
lines <- readLines(algo_script)
# Remove problematic install commands
lines <- lines[!grepl("BiocManager::install", lines)]
lines <- lines[!grepl("install.packages", lines)]
lines <- lines[!grepl("remotes::install", lines)]
lines <- lines[!grepl("devtools::install", lines)]
lines <- lines[!grepl("options\\(repos", lines)]
# Inject library loads for independence
lines <- c("library(stringr); library(dplyr); library(magrittr); library(tibble); library(tidyr)", lines)
writeLines(lines, algo_script)

# Signatures
sig_dir <- file.path(repo_dir, "scripts/deconvolution/signatures")
if (!dir.exists(sig_dir)) dir.create(sig_dir, recursive = TRUE)
lm22_file <- file.path(sig_dir, "LM22.txt")
if (!file.exists(lm22_file)) {
  download.file("https://raw.githubusercontent.com/mdozmorov/Immuno_notes/master/data/Cibersoft/LM22.txt", lm22_file)
}

# 3. Independent Execution Loop
# ------------------------------------------------------------------------------
# 3. Independent Execution Loop
# ------------------------------------------------------------------------------
print("[4/5] Preparing Input Files...")
data_dir <- file.path(getwd(), "TCGA_Real_data") # FORCE ABSOLUTE PATH
if (!dir.exists(data_dir)) stop(paste("TCGA_Real_data folder not found at:", data_dir))

all_files <- list.files(path = data_dir, pattern = "\\.(txt|tsv|csv)$", full.names = TRUE)
ignore_list <- c("deconvolution_results.csv", "input_matrix_merged.tsv")
input_files <- all_files[!basename(all_files) %in% ignore_list]

if (length(input_files) == 0) {
  stop("No input files found in TCGA_Real_data!")
}

print(paste(" -> Found", length(input_files), "patients to process independently."))

# ARGUMENT PARSING FOR PARALLEL EXECUTION
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 2) {
  start_idx <- as.integer(args[1])
  end_idx <- as.integer(args[2])
  suffix <- if (length(args) >= 3) args[3] else paste0(start_idx, "_", end_idx)
  print(paste("PARALLEL MODE: Processing files", start_idx, "to", end_idx, "Suffix:", suffix))
} else {
  start_idx <- 1
  end_idx <- length(input_files)
  suffix <- "FULL"
  print("SEQUENTIAL MODE: Processing all files")
}

# Slice input files
if (end_idx > length(input_files)) end_idx <- length(input_files)
if (start_idx <= end_idx) {
    input_files <- input_files[start_idx:end_idx]
} else {
    stop("Invalid start/end indices")
}

print(paste(" -> Handling subset of", length(input_files), "patients."))

# Switch to Repo WD for sourcing
current_wd <- getwd()
setwd(repo_dir)

# Load Functions
source("scripts/deconvolution/deconvolution_algorithms.R")
library(immunedeconv); library(EpiDISH); library(DeconRNASeq)

# Prepare Signature List Once
sig_files <- list.files("scripts/deconvolution/signatures", full.names = TRUE)
sig_files <- sig_files[!dir.exists(sig_files)] # Filter directories

results_list <- list()

print("[5/5] Starting Processing Loop...")
pb <- txtProgressBar(min = 0, max = length(input_files), style = 3)


for (i in seq_along(input_files)) {
  f <- input_files[i] # This is now GUARANTEED absolute
  # Extract Standard TCGA Barcode (TCGA-XX-XXXX-XXA)
  filename <- basename(f)
  if (grepl("TCGA-[0-9A-Z]{2}-[0-9A-Z]{4}-[0-9A-Z]{3}", filename)) {
    patient_id <- substr(filename, 1, 16)
  } else {
    patient_id <- tools::file_path_sans_ext(filename)
  }
  
  # Independent Processing Block
  local_res <- tryCatch({
    # 1. Read
    d <- suppressMessages(read_tsv(f, comment = "#", show_col_types = FALSE))
    if(ncol(d) < 2) d <- suppressMessages(read_csv(f, show_col_types = FALSE))
    
    # 2. Standardize
    if ("gene_name" %in% colnames(d)) {
      d <- d %>% select(Gene = gene_name, Value = tpm_unstranded)
    } else {
      d <- d %>% select(Gene = 1, Value = 2)
    }
    
    # 3. Create Patient Matrix (Single Column)
    mx <- d %>%
      filter(!is.na(Gene)) %>%
      mutate(Value = as.numeric(Value)) %>%
      group_by(Gene) %>%
      summarise(Value = mean(Value, na.rm=TRUE), .groups = "drop") %>%
      column_to_rownames("Gene") %>%
      as.matrix()
    colnames(mx) <- patient_id
    mx[is.na(mx)] <- 0
    
    # --- DIMENSION HACK ---
    # Quantiseq/EpiDISH fail on single-column matrices. We add a dummy duplicate.
    mx_run <- cbind(mx, mx)
    colnames(mx_run) <- c(patient_id, paste0(patient_id, "_Dummy"))
    
    # 4. Run Algorithms
    # Quantiseq
    print(paste("Running Quantiseq for:", patient_id))
    res_q <- tryCatch({
        r <- computeQuantiseq(mx_run)
        if(is.null(r) || nrow(r) == 0) stop("Quantiseq returned NULL/Empty")
        r <- r %>% filter(sample == patient_id) # Keep only real
        print(paste("  -> Quantiseq cols:", paste(colnames(r), collapse=",")))
        r
    }, error=function(e) {
        print(paste("  [ERROR] Quantiseq failed:", e$message))
        tibble(sample=patient_id) 
    })
    
    # EpiDISH
    print(paste("Running EpiDISH for:", patient_id))
    res_v <- tryCatch({
        r <- methods_with_variable_signatures(mx_run, sig_files)
        if(is.null(r) || nrow(r) == 0) stop("EpiDISH returned NULL/Empty")
        r <- r %>% filter(sample == patient_id) # Keep only real
        print(paste("  -> EpiDISH cols:", paste(colnames(r), collapse=",")))
        r
    }, error=function(e) {
        print(paste("  [ERROR] EpiDISH failed:", e$message))
        tibble(sample=patient_id)
    })
    
    # 5. Merge Result
    # Normalize 'sample' column name just in case
    if("Samples" %in% colnames(res_q)) res_q <- res_q %>% rename(sample = Samples)
    if("Samples" %in% colnames(res_v)) res_v <- res_v %>% rename(sample = Samples)
    
    combined <- full_join(res_q, res_v, by="sample")
    print(paste("  -> Combined cols:", paste(colnames(combined), collapse=",")))
    combined

  }, error = function(e) {
    print(paste("CRITICAL FILE ERROR:", e$message))
    tibble(sample=patient_id, Error=paste("File Read Error:", e$message))
  })
  
  results_list[[i]] <- local_res
  setTxtProgressBar(pb, i)
}

close(pb)
setwd(current_wd) # Restore WD

# 4. Aggregate & Save
# ------------------------------------------------------------------------------
print("Aggregating results...")
final_df <- bind_rows(results_list)

output_file <- paste0("all_samples_GEMDeCan_Independent_results_", suffix, ".csv")
write_csv(final_df, output_file)

print("=======================================================")
print(paste("DONE! Results saved for", length(results_list), "patients."))
print(paste("Output file:", output_file))
print("=======================================================")
print(head(final_df))
