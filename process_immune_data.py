import pandas as pd
import numpy as np

def process_immune_data():
    # File paths
    deconv_file = 'all_samples_GEMDeCan_Independent_results.csv'
    mitcr_file = 'mitcr_sampleStatistics_20160714.tsv'
    output_file = 'merged_immune_features.csv'

    print("Loading data files...")
    # Load all partial results
    import glob
    import os
    
    pattern = 'all_samples_GEMDeCan_Independent_results*.csv'
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern}")
        
    print(f"Found {len(files)} result files: {files}")
    deconv_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    mitcr_df = pd.read_csv(mitcr_file, sep='\t')

    print(f"Deconvolution data shape: {deconv_df.shape}")
    print(f"TCR data shape: {mitcr_df.shape}")

    # --- Step 1: The Clean & Merge ---

    print("Processing TCR data...")
    # Create Patient_ID from AliquotBarcode (first 12 chars)
    # The clean ID format is 'TCGA-XX-XXXX'
    # Check if 'AliquotBarcode' exists, otherwise look for other barcode columns
    barcode_col = 'AliquotBarcode' if 'AliquotBarcode' in mitcr_df.columns else mitcr_df.columns[0]
    
    mitcr_df['Patient_ID'] = mitcr_df[barcode_col].astype(str).str.slice(0, 12)
    
    # We might have potential duplicates if multiple samples exist for one patient in TCR data
    # The user didn't specify how to handle duplicates, but taking the mean or first is reasonable.
    # However, for simplicity and to avoid data loss, we'll keep duplicates for now or maybe just drop them if they are identical?
    # Extract Sample ID from TCGA barcode (first 16 characters, e.g., TCGA-XX-XXXX-XXA)
    # The deconvolution results use Sample ID (16 chars), so we must match on that.
    mitcr_df['Sample_ID'] = mitcr_df[barcode_col].astype(str).str.slice(0, 16)
    
    # Select only necessary columns and drop duplicates
    mitcr_clean = mitcr_df[['Sample_ID', 'shannon']].drop_duplicates(subset=['Sample_ID'])
    
    print(f"Loaded TCR data: {len(mitcr_clean)} unique samples.")
    print(f"Sample ID examples from TCR data: {mitcr_clean['Sample_ID'].head().tolist()}")
    print(f"Sample ID examples from Deconv data: {deconv_df['sample'].head().tolist()}")

    # Merge deconvolution results with TCR data
    merged_df = deconv_df.merge(mitcr_clean, left_on='sample', right_on='Sample_ID', how='left')
    
    print(f"Merged data shape: {merged_df.shape}")
    print(f"Number of samples with valid Shannon score: {merged_df['shannon'].notna().sum()}")

    # Fill NaN values in 'shannon' with 0 (assuming missing means no diversity/data)
    merged_df['shannon'] = merged_df['shannon'].fillna(0)



    # --- Step 2: Feature Engineering ---
    
    print("Performing feature engineering...")
    
    # 1. Immune_Engagement_Index = Quantiseq_T_cell_CD8+ / Shannon_Entropy
    # Handle division by zero: if shannon is 0, the result is inf. 
    # We can replace inf with 0 or a large number, or leave as np.inf. 
    # User logic: "How many T-cells are there relative to their diversity?"
    # If diversity is 0 (no clonality), but there are T cells, it's weird. 
    # If shannon is 0, it means numClones is 0 or 1.
    # Let's use a safe division that produces np.inf or NaN, but maybe 0 is safer if user filled 0.
    # If I respect the math: x / 0 = inf.
    # Let's add a small epsilon to avoid error, or just let pandas handle it (it produces inf).
    # I will stick to the formula.
    
    # Check for exact column names
    cd8_col = 'Quantiseq_T_cell_CD8+'
    m2_col = 'Quantiseq_Macrophage_M2'
    treg_col = 'Quantiseq_T_cell_regulatory_(Tregs)'
    
    if cd8_col not in merged_df.columns:
        print(f"Warning: {cd8_col} not found!")
    if m2_col not in merged_df.columns:
        print(f"Warning: {m2_col} not found!")
    if treg_col not in merged_df.columns:
        print(f"Warning: {treg_col} not found!")

    # Avoid DivisionByZero error by masking or using numpy
    # merged_df['Immune_Engagement_Index'] = merged_df[cd8_col] / merged_df['shannon']
    # If shannon is 0, we get inf.
    
    merged_df['Immune_Engagement_Index'] = merged_df.apply(
        lambda row: row[cd8_col] / row['shannon'] if row['shannon'] > 0 else 0, axis=1
    )
    # Rationale: If shannon is 0 (no clonality), user said "fill with 0". 
    # If diversity is 0, we can argue "Engagement" is undefined or 0. Setting to 0 is a safe default for ML/Analysis 
    # rather than Inf which breaks models. 

    # 2. Macrophage_Blockade = Quantiseq_Macrophage_M2 * Quantiseq_T_cell_regulatory_(Tregs)
    merged_df['Macrophage_Blockade'] = merged_df[m2_col] * merged_df[treg_col]

    # --- Reorder Columns ---
    print("Reordering columns...")
    immune_cols = ['shannon', 'Immune_Engagement_Index', 'Macrophage_Blockade']
    # Ensure 'sample' is first, then immune cols, then rest
    base_cols = ['sample']
    # Get all other cols that are not in immune/base
    other_cols = [c for c in merged_df.columns if c not in base_cols + immune_cols]
    
    final_cols = base_cols + immune_cols + other_cols
    merged_df = merged_df[final_cols]

    # --- Save ---
    print(f"Saving results to {output_file}...")
    merged_df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    process_immune_data()
