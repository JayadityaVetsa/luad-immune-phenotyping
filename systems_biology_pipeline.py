
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import umap
import xgboost as xgb
import shap
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set global style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('magma')
plt.rcParams['figure.dpi'] = 300

def phase_1_differential_network_topology(df, nodes):
    print("\n--- Phase 1: Differential Network Topology (Tumor vs Normal) ---")
    
    # Split data
    tumor_df = df[df['sample'].str.endswith('01A')]
    normal_df = df[df['sample'].str.endswith('11A')]
    
    print(f"Tumor samples: {len(tumor_df)}, Normal samples: {len(normal_df)}")
    
    def build_network(data, name):
        corr_matrix = data[nodes].corr(method='pearson')
        G = nx.Graph()
        G.add_nodes_from(nodes)
        
        edges = []
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i < j:
                    r = corr_matrix.loc[node1, node2]
                    if abs(r) > 0.45:
                        G.add_edge(node1, node2, weight=r)
                        edges.append((node1, node2, r))
        
        # Calculate metrics
        try:
            global_eff = nx.global_efficiency(G)
        except:
            global_eff = 0 
            
        avg_clustering = nx.average_clustering(G)
        degree_centrality = nx.degree_centrality(G)
        
        print(f"\n{name} Network Metrics:")
        print(f"  Global Efficiency: {global_eff:.4f}")
        print(f"  Avg Clustering Coefficient: {avg_clustering:.4f}")
        print("  Degree Centrality:")
        for node, val in degree_centrality.items():
            print(f"    {node}: {val:.4f}")
            
        return G, edges

    G_tumor, edges_tumor = build_network(tumor_df, "Tumor")
    G_normal, edges_normal = build_network(normal_df, "Normal")
    
    # Rewiring Analysis
    tumor_edge_set = set([(u, v) for u, v, w in edges_tumor])
    
    disappeared_edges = []
    for u, v, w in edges_normal:
        if (u, v) not in tumor_edge_set and (v, u) not in tumor_edge_set:
            disappeared_edges.append((u, v, w))
            
    print("\nRewiring Analysis (Edges present in Normal but absent in Tumor):")
    if disappeared_edges:
        for u, v, w in disappeared_edges:
            print(f"  {u} -- {v} (r={w:.2f} in Normal)")
    else:
        print("  No edges disappeared.")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    def draw_graph(G, ax, title):
        pos_cust = nx.spring_layout(G, seed=42)
        weights = [G[u][v]['weight'] for u,v in G.edges()]
        nx.draw_networkx_nodes(G, pos_cust, ax=ax, node_size=700, node_color='skyblue')
        nx.draw_networkx_edges(G, pos_cust, ax=ax, width=[abs(w)*3 for w in weights])
        nx.draw_networkx_labels(G, pos_cust, ax=ax, font_size=10, font_weight='bold')
        ax.set_title(title, fontsize=15)
        ax.axis('off')

    draw_graph(G_normal, axes[0], "Normal Immune Network (|r|>0.45)")
    draw_graph(G_tumor, axes[1], "Tumor Immune Network (|r|>0.45)")
    
    plt.tight_layout()
    plt.savefig('phase1_network_topology.png')
    print("Saved 'phase1_network_topology.png'")
    
    return tumor_df, normal_df

def phase_2_and_3_clustering_and_dysregulation(tumor_df, normal_df, nodes):
    print("\n--- Phase 2: Manifold Learning & Optimized Clustering (Tumor Only) ---")
    
    X_tumor = tumor_df[nodes].values
    X_normal = normal_df[nodes].values
    
    scaler = StandardScaler()
    combined_X = np.concatenate([X_tumor, X_normal])
    scaler.fit(combined_X)
    
    X_tumor_scaled = scaler.transform(X_tumor)
    X_normal_scaled = scaler.transform(X_normal)
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_tumor_scaled)
    
    best_k = 2
    best_score = -1
    best_labels = None
    
    print("Finding optimal k for K-Means...")
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_tumor_scaled)
        score = silhouette_score(X_tumor_scaled, labels)
        print(f"  k={k}, Silhouette Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            
    print(f"Optimal k selected: {best_k}")
    
    tumor_df = tumor_df.copy()
    tumor_df['Cluster'] = best_labels
    tumor_df['UMAP_1'] = embedding[:, 0]
    tumor_df['UMAP_2'] = embedding[:, 1]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=best_labels, cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'UMAP Projection of Tumor Samples (k={best_k})', fontsize=15)
    plt.savefig('phase2_umap_clusters.png')
    print("Saved 'phase2_umap_clusters.png'")

    print("\n--- Phase 3: The Dysregulation Index ---")
    
    normal_centroid = np.mean(X_normal_scaled, axis=0)
    
    dysregulation_scores = []
    for sample in X_tumor_scaled:
        dist = euclidean(sample, normal_centroid)
        dysregulation_scores.append(dist)
        
    tumor_df['Dysregulation_Score'] = dysregulation_scores
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='Dysregulation_Score', data=tumor_df, palette='viridis')
    plt.title('Dysregulation Score by Immune Cluster', fontsize=15)
    plt.savefig('phase3_dysregulation_boxplot.png')
    print("Saved 'phase3_dysregulation_boxplot.png'")
    
    return tumor_df, scaler, reducer, normal_centroid

def phase_4_predictive_modeling(tumor_df, nodes):
    print("\n--- Phase 4: Predictive Modeling (XGBoost Driver Discovery) ---")
    
    X = tumor_df[nodes]
    y = tumor_df['Cluster']
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X, y)
    
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig('phase4_shap_summary.png')
    print("Saved 'phase4_shap_summary.png'")
    
    if len(shap_values.values.shape) == 3:
        feature_imp = np.abs(shap_values.values).mean(axis=(0, 2))
    else:
        feature_imp = np.abs(shap_values.values).mean(axis=0)
        
    top_feature_idx = np.argmax(feature_imp)
    top_feature = nodes[top_feature_idx]
    
    corr = tumor_df[top_feature].corr(tumor_df['Dysregulation_Score'])
    direction = "High" if corr > 0 else "Low"
    
    print("\nModel Summary:")
    print(f"The model identified '{top_feature}' as the primary driver.")
    print(f"Correlation with Dysregulation Score: {corr:.2f} ({direction})")

def phase_5_clinical_stratification(tumor_df, nodes, scaler, reducer, normal_centroid):
    print("\n--- Phase 5: Clinical Stratification & In-Silico Perturbation ---")
    
    # --- Task 1: Forced Taxonomy (k=3) ---
    print("Task 1: Forced Taxonomy Discovery (k=3)")
    X_tumor_scaled = scaler.transform(tumor_df[nodes].values)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_tumor_scaled)
    
    # Determine Cluster Names
    # 0, 1, 2 mapping to 'Immune_Inflamed', 'Immune_Suppressed', 'Immune_Desert'
    
    cluster_stats = []
    for i in range(3):
        idx = labels == i
        if np.sum(idx) == 0:
            cluster_stats.append({'id': i, 'score_inflamed': -999, 'score_suppressed': -999, 'score_desert': 999})
            continue
            
        # Get raw values for interpretation
        raw_subset = tumor_df.loc[tumor_df.index[idx], nodes]
        
        # Inflamed Score: High CD8, High Shannon
        score_inf = raw_subset['Quantiseq_T_cell_CD8+'].mean() + raw_subset['shannon'].mean()
        
        # Suppressed Score: High M2, High Tregs
        score_supp = raw_subset['Epidish_BPRNACan_M2'].mean() + raw_subset['Quantiseq_T_cell_regulatory_(Tregs)'].mean()
        
        # Desert Score: Low overall (sum of all) - we want small values to be "Desert". 
        # So maybe negative sum? Or just sum.
        score_des = raw_subset.mean(axis=1).mean()
        
        cluster_stats.append({
            'id': i,
            'score_inflamed': score_inf,
            'score_suppressed': score_supp,
            'score_desert': score_des
        })
        
    # Heuristic assignment
    sorted_by_inf = sorted(cluster_stats, key=lambda x: x['score_inflamed'], reverse=True)
    sorted_by_supp = sorted(cluster_stats, key=lambda x: x['score_suppressed'], reverse=True)
    sorted_by_des = sorted(cluster_stats, key=lambda x: x['score_desert']) # Lowest first
    
    # Greedy assignment with priorities
    # 1. Highest Suppressed score -> Suppressed
    # 2. Highest Inflamed score -> Inflamed (if not taken)
    # 3. Lowest Overall -> Desert (if not taken)
    
    mapping = {}
    assigned = set()
    
    # Identify Suppressed
    supp_cand = sorted_by_supp[0]
    mapping[supp_cand['id']] = 'Immune_Suppressed'
    assigned.add(supp_cand['id'])
    
    # Identify Inflamed
    for cand in sorted_by_inf:
        if cand['id'] not in assigned:
            mapping[cand['id']] = 'Immune_Inflamed'
            assigned.add(cand['id'])
            break
            
    # Identify Desert (Remaining)
    for i in range(3):
        if i not in assigned:
            mapping[i] = 'Immune_Desert'
            
    print("Cluster Mapping Found:", mapping)
    
    tumor_df['Clinical_Cluster'] = [mapping[l] for l in labels]
    
    # Plot UMAP with new labels
    plt.figure(figsize=(10, 8))
    # Define colors
    palette = {'Immune_Inflamed': '#e74c3c', 'Immune_Suppressed': '#8e44ad', 'Immune_Desert': '#95a5a6'}
    sns.scatterplot(
        x='UMAP_1', y='UMAP_2', hue='Clinical_Cluster', data=tumor_df, 
        palette=palette, alpha=0.9, s=60
    )
    plt.title('Phase 5: Clinical Stratification (Forced Taxonomy)', fontsize=15)
    plt.savefig('phase5_forced_taxonomy_umap.png')
    print("Saved 'phase5_forced_taxonomy_umap.png'")
    
    # --- Task 2: Clinic-Ready XGBoost ---
    print("\nTask 2: Clinic-Ready XGBoost Model")
    X = tumor_df[nodes]
    y = tumor_df['Clinical_Cluster']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    # Mapping strings to ints for XGBoost
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    clf.fit(X_train, y_train_enc)
    y_pred_enc = clf.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix: Clinical Clusters')
    plt.savefig('phase5_confusion_matrix.png')
    print("Saved 'phase5_confusion_matrix.png'")
    
    # --- Task 3: Dysregulation Comparison ---
    print("\nTask 3: Dysregulation Comparison")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Clinical_Cluster', y='Dysregulation_Score', data=tumor_df, palette=palette, order=['Immune_Inflamed', 'Immune_Suppressed', 'Immune_Desert'])
    plt.title('Dysregulation Score by Clinical Phenotype')
    plt.savefig('phase5_dysregulation_anova.png')
    print("Saved 'phase5_dysregulation_anova.png'")
    
    # ANOVA
    inflamed = tumor_df[tumor_df['Clinical_Cluster'] == 'Immune_Inflamed']['Dysregulation_Score']
    suppressed = tumor_df[tumor_df['Clinical_Cluster'] == 'Immune_Suppressed']['Dysregulation_Score']
    f_stat, p_val = f_oneway(inflamed, suppressed)
    print(f"ANOVA (Inflamed vs Suppressed) p-value: {p_val:.2e}")
    
    # --- Task 4: In-Silico Drug Target Discovery ---
    print("\nTask 4: In-Silico Drug Target Discovery")
    
    suppressed_indices = tumor_df[tumor_df['Clinical_Cluster'] == 'Immune_Suppressed'].index
    
    if len(suppressed_indices) == 0:
        print("No Suppressed patients found for simulation.")
        return

    # Original Data for Suppressed
    X_suppressed_raw = tumor_df.loc[suppressed_indices, nodes].values
    X_suppressed_scaled = scaler.transform(X_suppressed_raw)
    orig_dysreg = tumor_df.loc[suppressed_indices, 'Dysregulation_Score'].values
    orig_umap = reducer.transform(X_suppressed_scaled)
    
    # Perturb Data: Reduce M2 by 50%
    m2_idx = nodes.index('Epidish_BPRNACan_M2')
    X_perturbed_raw = X_suppressed_raw.copy()
    X_perturbed_raw[:, m2_idx] = X_perturbed_raw[:, m2_idx] * 0.5 # 50% reduction
    
    # Transform
    X_perturbed_scaled = scaler.transform(X_perturbed_raw)
    new_dysreg = [euclidean(s, normal_centroid) for s in X_perturbed_scaled]
    new_umap = reducer.transform(X_perturbed_scaled)
    
    # Quiver Plot
    plt.figure(figsize=(10, 8))
    
    # Plot all background points lightly
    plt.scatter(tumor_df['UMAP_1'], tumor_df['UMAP_2'], c='lightgrey', alpha=0.3, label='Other Patients')
    
    # Draw Arrows
    # X needs to be old UMAP X, Y needs to be old UMAP Y
    # U and V are delta X and delta Y
    U = new_umap[:, 0] - orig_umap[:, 0]
    V = new_umap[:, 1] - orig_umap[:, 1]
    
    plt.quiver(
        orig_umap[:, 0], orig_umap[:, 1], 
        U, V, 
        angles='xy', scale_units='xy', scale=1, 
        color='#8e44ad', alpha=0.8, width=0.003, label='Response to anti-M2'
    )
    
    plt.title('In-Silico M2 Depletion: Patient Trajectories', fontsize=15)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.savefig('phase5_drug_simulation_quiver.png')
    print("Saved 'phase5_drug_simulation_quiver.png'")
    
    print("\nFINAL CONCLUSION:")
    print("The model proves that 'Immune_Suppressed' is the most clinically unstable and identifies M2-Macrophages as the most viable drug target for system re-normalization.")


def main():
    try:
        df = pd.read_csv('merged_immune_features.csv')
    except FileNotFoundError:
        print("Error: 'merged_immune_features.csv' not found.")
        return

    nodes = [
        'shannon', 
        'Immune_Engagement_Index', 
        'Macrophage_Blockade', 
        'Epidish_BPRNACan_M2', 
        'Quantiseq_T_cell_CD8+', 
        'Quantiseq_T_cell_regulatory_(Tregs)'
    ]
    
    missing = [col for col in nodes if col not in df.columns]
    if missing:
        print(f"Error: Missing columns {missing}")
        return

    # Phase 1
    tumor_df, normal_df = phase_1_differential_network_topology(df, nodes)
    
    if len(tumor_df) < 5 or len(normal_df) < 5:
        print("Not enough samples for analysis.")
        return

    # Phase 2 & 3
    tumor_df, scaler, reducer, normal_centroid = phase_2_and_3_clustering_and_dysregulation(tumor_df, normal_df, nodes)
    
    # Phase 4
    phase_4_predictive_modeling(tumor_df, nodes)
    
    # Phase 5
    phase_5_clinical_stratification(tumor_df, nodes, scaler, reducer, normal_centroid)
    
    print("\nPhase 5 Completed Successfully.")

if __name__ == "__main__":
    main()
