
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import f_oneway
import glob
import json
import os
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('magma')
plt.rcParams['figure.dpi'] = 300

# Constants
NODES = [
    'shannon', 
    'Immune_Engagement_Index', 
    'Macrophage_Blockade', 
    'Epidish_BPRNACan_M2', 
    'Quantiseq_T_cell_CD8+', 
    'Quantiseq_T_cell_regulatory_(Tregs)'
]
CLINICAL_DIR = 'patient_clinical_data'
TCGA_DIR = 'TCGA_Real_data'

def load_and_restore_scores(df):
    """
    Recover Phase 5 scores (Dysregulation, Clinical_Cluster) 
    by re-running the logic on the current feature set.
    """
    print("Evaluating Phase 5 Logic to restore scores...")
    
    # Filter Tumor and Normal
    tumor_df = df[df['sample'].str.endswith('01A')].copy()
    normal_df = df[df['sample'].str.endswith('11A')].copy()
    
    # Scaling
    scaler = StandardScaler()
    combined_X = np.concatenate([tumor_df[NODES].values, normal_df[NODES].values])
    scaler.fit(combined_X)
    
    X_tumor_scaled = scaler.transform(tumor_df[NODES].values)
    X_normal_scaled = scaler.transform(normal_df[NODES].values)
    normal_centroid = np.mean(X_normal_scaled, axis=0)
    
    # Dysregulation Score
    tumor_df['Dysregulation_Score'] = [euclidean(s, normal_centroid) for s in X_tumor_scaled]
    
    # Forced Taxonomy (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_tumor_scaled)
    
    # Naming Logic
    cluster_stats = []
    for i in range(3):
        idx = labels == i
        if np.sum(idx) == 0:
             cluster_stats.append({'id': i, 'inf': -999, 'supp': -999, 'des': 999})
             continue
        subset = tumor_df.iloc[idx]
        inf = subset['Quantiseq_T_cell_CD8+'].mean() + subset['shannon'].mean()
        supp = subset['Epidish_BPRNACan_M2'].mean() + subset['Quantiseq_T_cell_regulatory_(Tregs)'].mean()
        des = subset[NODES].mean(axis=1).mean()
        cluster_stats.append({'id': i, 'inf': inf, 'supp': supp, 'des': des})
        
    mapping = {}
    assigned = set()
    
    # Suppressed (Highest Supp score)
    s_c = sorted(cluster_stats, key=lambda x: x['supp'], reverse=True)[0]
    mapping[s_c['id']] = 'Immune_Suppressed'
    assigned.add(s_c['id'])
    
    # Inflamed (Highest Inf score, not assigned)
    inf_sorted = sorted(cluster_stats, key=lambda x: x['inf'], reverse=True)
    for c in inf_sorted:
        if c['id'] not in assigned:
            mapping[c['id']] = 'Immune_Inflamed'
            assigned.add(c['id'])
            break
            
    # Desert (Remaining)
    for i in range(3):
        if i not in assigned:
            mapping[i] = 'Immune_Desert'
            
    tumor_df['Clinical_Cluster'] = [mapping[l] for l in labels]
    
    return tumor_df, scaler, normal_centroid

def task_1_clinical_survival(tumor_df):
    print("\n--- Task 1: Clinical Survival Integration ---")
    
    clinical_data = []
    json_files = glob.glob(os.path.join(CLINICAL_DIR, '*.json'))
    print(f"Found {len(json_files)} clinical JSON files.")
    
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8-sig') as f:
                content = f.read().strip()
                if not content:
                    print(f"Skipping empty file: {jf}")
                    continue
                data = json.loads(content)
        except Exception as e:
            print(f"Skipping corrupt file {jf}: {e}")
            continue
            
        # Parse ID from filename
        fname = os.path.basename(jf)
        sample_id = fname.replace("_clinical.json", "") 
        
        try:
            vital = data.get('demographic', {}).get('vital_status')
            age = data.get('demographic', {}).get('age_at_index')
            
            # Survival Time
            diag = data.get('diagnoses', [{}])[0]
            if vital == 'Dead':
                event = 1
                time = diag.get('days_to_death')
                if time is None: 
                     time = diag.get('days_to_last_follow_up')
            else:
                event = 0
                time = diag.get('days_to_last_follow_up')
                
            if time is None or age is None:
                print(f"Skipping {sample_id}: Missing Time ({time}) or Age ({age})")
                continue
                
            clinical_data.append({
                'sample': sample_id,
                'Event': event,
                'Time': float(time),
                'Age': float(age)
            })
            
        except Exception as e:
            print(f"Error parsing {sample_id}: {e}")
            continue
    
    print(f"Extracted {len(clinical_data)} clinical records.")

            
    clinical_df = pd.DataFrame(clinical_data)
    
    # Clean Columns
    tumor_df.columns = tumor_df.columns.str.strip()
    
    # Debug
    print("Tumor Columns List:", list(tumor_df.columns))
    
    # Ensure types
    if 'sample' in tumor_df.columns:
        tumor_df['sample'] = tumor_df['sample'].astype(str)
        
    clinical_df = pd.DataFrame(clinical_data)
    if 'sample' in clinical_df.columns:
        clinical_df['sample'] = clinical_df['sample'].astype(str)
        
    print("Clinical Columns List:", list(clinical_df.columns))
    
    # Merge
    try:
        merged_surv = pd.merge(tumor_df, clinical_df, on='sample', how='inner')
    except KeyError as e:
        print(f"Merge failed: {e}")
        print("Tumor index name:", tumor_df.index.name)
        return tumor_df, None

    print(f"Matched {len(merged_surv)} patients with clinical data.")
    
    if len(merged_surv) < 10:
        print("Not enough data for Cox Model.")
        return merged_surv, None

    cph = CoxPHFitter()
    # Normalize Covariates for better convergence
    surv_subset = merged_surv[['Time', 'Event', 'Dysregulation_Score', 'Age']]
    cph.fit(surv_subset, duration_col='Time', event_col='Event')
    
    print("\nCox Proportional Hazards Model Results:")
    cph.print_summary()
    
    # Forest Plot
    plt.figure(figsize=(8, 6))
    cph.plot()
    plt.title('Survival Forest Plot (Hazard Ratios)')
    plt.tight_layout()
    plt.savefig('capstone_survival_forest.png')
    print("Saved 'capstone_survival_forest.png'")
    
    return merged_surv, cph

def task_2_paracrine_crosstalk(tumor_df):
    print("\n--- Task 2: Paracrine Crosstalk Modeling ---")
    
    gene_map = {}
    
    # Iterate through matched patients in tumor_df
    # We need to find corresponding TSV files. 
    # TSV Filename: sample.tsv (e.g., TCGA-38-4625-01A.tsv)
    
    for idx, row in tumor_df.iterrows():
        sample = row['sample']
        tsv_path = os.path.join(TCGA_DIR, f"{sample}.tsv")
        
        if not os.path.exists(tsv_path):
            continue
            
        try:
            # Read TSV. Header is usually line 2 (1-indexed) -> skiprows=1
            # Checking view_file output: line 2 has 'gene_id gene_name...'
            # So header=1 (0-indexed) or just read_csv with sep='\t', header=1
            
            # Using basic manual read to be robust or pandas
            tpm_df = pd.read_csv(tsv_path, sep='\t', header=1) 
            
            # Extract Genes of Interest
            target_genes = ['TGFB1', 'IL10', 'IFNG']
            subset = tpm_df[tpm_df['gene_name'].isin(target_genes)]
            
            vals = {g: 0.0 for g in target_genes}
            for _, g_row in subset.iterrows():
                vals[g_row['gene_name']] = float(g_row['tpm_unstranded'])
                
            gene_map[sample] = vals
            
        except Exception as e:
            # print(f"Error reading {sample}: {e}")
            continue
            
    # Calculate Fluxes
    crosstalk_data = []
    for idx, row in tumor_df.iterrows():
        sample = row['sample']
        if sample not in gene_map:
            crosstalk_data.append({'Suppressive_Flux': np.nan, 'Effector_Flux': np.nan, 'Crosstalk_Balance_Ratio': np.nan})
            continue
            
        g = gene_map[sample]
        m2 = row['Epidish_BPRNACan_M2']
        cd8 = row['Quantiseq_T_cell_CD8+']
        
        supp_flux = m2 * (g['TGFB1'] + g['IL10'])
        eff_flux = cd8 * g['IFNG']
        cbr = eff_flux / (supp_flux + 1e-9)
        
        crosstalk_data.append({
            'Suppressive_Flux': supp_flux,
            'Effector_Flux': eff_flux, 
            'Crosstalk_Balance_Ratio': cbr
        })
        
    crosstalk_df = pd.DataFrame(crosstalk_data, index=tumor_df.index)
    tumor_df = pd.concat([tumor_df, crosstalk_df], axis=1)
    
    return tumor_df

def task_3_therapeutic_plasticity_digital_twin(tumor_df, scaler, normal_centroid):
    print("\n--- Task 3: Therapeutic Plasticity (Digital Twin) ---")
    
    # Target: Immune_Suppressed
    target_idx = tumor_df[tumor_df['Clinical_Cluster'] == 'Immune_Suppressed'].index
    
    plasticity_scores = []
    recommendations = []
    
    # Feature indices for perturbation
    # NODES order: 'shannon', 'Immune_Engagement_Index', 'Macrophage_Blockade', 'Epidish_BPRNACan_M2', 'Quantiseq_T_cell_CD8+', 'Quantiseq_T_cell_regulatory_(Tregs)'
    m2_idx = NODES.index('Epidish_BPRNACan_M2')
    cd8_idx = NODES.index('Quantiseq_T_cell_CD8+')

    for idx in tumor_df.index:
        if idx not in target_idx:
            plasticity_scores.append(np.nan)
            
            # Default logic for others
            cluster = tumor_df.loc[idx, 'Clinical_Cluster']
            if cluster == 'Immune_Inflamed':
                recommendations.append('Standard_Immunotherapy')
            elif cluster == 'Immune_Desert':
                recommendations.append('Priming_Therapy')
            else:
                 recommendations.append('Observation') # Should catch Suppressed in the other branch
            continue
            
        # Digital Twin Simulation
        original_vals = tumor_df.loc[idx, NODES].values.copy().astype(float)
        original_dysreg = tumor_df.loc[idx, 'Dysregulation_Score']
        
        # Perturb
        new_vals = original_vals.copy()
        new_vals[m2_idx] *= 0.5   # Decrease M2 by 50%
        new_vals[cd8_idx] *= 1.5  # Increase CD8 by 50%
        
        # Recalculate Score
        # Transform using saved scaler
        new_vals_scaled = scaler.transform(new_vals.reshape(1, -1))
        new_dysreg = euclidean(new_vals_scaled[0], normal_centroid)
        
        # Score
        pct_improvement = ((original_dysreg - new_dysreg) / original_dysreg) * 100
        plasticity_scores.append(pct_improvement)
        
        if pct_improvement > 10:
            recommendations.append('Targeted_M2_Depletion') # "Systemically Plastic" -> Good candidate for this therapy
        else:
            recommendations.append('Combination_Therapy') # "Systemically Resilient" -> Needs more aggressive/combo
            
    tumor_df['Plasticity_Score'] = plasticity_scores
    tumor_df['Therapeutic_Recommendation'] = recommendations
    
    return tumor_df

def main():
    try:
        df = pd.read_csv('merged_immune_features.csv')
    except FileNotFoundError:
        print("Error: 'merged_immune_features.csv' not found.")
        return

    # Check Columns
    missing = [c for c in NODES if c not in df.columns]
    if missing:
        print("Missing columns:", missing)
        return

    # Restore Phase 5 State
    tumor_df, scaler, normal_centroid = load_and_restore_scores(df)
    
    # Task 1: Survival
    tumor_df, cph = task_1_clinical_survival(tumor_df)
    if cph is None:
        print("\nSkipping Tasks needing clinical data due to match failure.")
        return

    # Task 2: Crosstalk
    tumor_df = task_2_paracrine_crosstalk(tumor_df)
    
    # Task 3: Digital Twin
    tumor_df = task_3_therapeutic_plasticity_digital_twin(tumor_df, scaler, normal_centroid)
    
    # Task 4: Visualization & Output
    print("\n--- Task 4: Visualization & Export ---")
    
    # Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Clinical_Cluster', y='Crosstalk_Balance_Ratio', data=tumor_df, palette='magma', order=['Immune_Inflamed', 'Immune_Suppressed', 'Immune_Desert'])
    plt.title('Paracrine Crosstalk Balance Ratio (CBR) by Cluster')
    plt.ylabel('CBR (Effector / Suppressive Flux)')
    plt.yscale('log') # CBR might span orders of magnitude
    plt.savefig('capstone_crosstalk_violin.png')
    print("Saved 'capstone_crosstalk_violin.png'")
    
    # Export CSV
    out_cols = ['sample', 'Clinical_Cluster', 'Dysregulation_Score', 'Crosstalk_Balance_Ratio', 'Plasticity_Score', 'Therapeutic_Recommendation']
    tumor_df[out_cols].to_csv('Final_Clinical_Insight_Table.csv', index=False)
    print("Saved 'Final_Clinical_Insight_Table.csv'")
    
    print("\nCapstone Pipeline Completed Successfully.")
    
if __name__ == "__main__":
    main()
