
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6), 'figure.dpi': 100})

def prepare_data(filepath):
    """
    Step 1: Data Preparation & Labeling
    """
    print("Step 1: Loading and Preparing Data...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None

    # Create Label column
    # If the barcode contains "-01A", Label = 1 (Tumor).
    # If the barcode contains "-11A", Label = 0 (Normal).
    
    def get_label(sample_id):
        if "-01A" in str(sample_id):
            return 1
        elif "-11A" in str(sample_id):
            return 0
        else:
            return np.nan

    df['Label'] = df['sample'].apply(get_label)
    
    # Drop rows where Label is NaN (if any samples don't match criteria)
    df = df.dropna(subset=['Label'])
    df['Label'] = df['Label'].astype(int)

    # Sanity Check
    print("Sample Count:")
    print(df['Label'].value_counts().rename({1: 'Tumor (-01A)', 0: 'Normal (-11A)'}))

    # Drop non-numeric columns and the original sample column
    # Keeping only numeric features and Label
    # We need to exclude 'sample' and maybe 'Sample_ID' if it exists differently
    
    # Identify non-numeric columns to drop (including 'sample' and 'Sample_ID')
    cols_to_drop = ['sample']
    if 'Sample_ID' in df.columns:
        cols_to_drop.append('Sample_ID')
        
    # Also ignore any other potential string columns just in case
    for col in df.select_dtypes(include=['object']).columns:
        if col not in cols_to_drop:
             cols_to_drop.append(col)
    
    # Filter out cancer/malignant cell features to focus on Immune Signatures
    # This prevents the model from just seeing "Cancer cells > 0" -> Tumor
    # and ensures we are looking at the immune microenvironment.
    for col in df.columns:
        col_lower = col.lower()
        if 'cancer' in col_lower or 'malignant' in col_lower or 'tumor' in col_lower:
            cols_to_drop.append(col)
             
    X = df.drop(columns=cols_to_drop + ['Label'])
    y = df['Label']
    
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    return X, y

def train_model(X, y):
    """
    Step 2: Model Training
    """
    print("\nStep 2: Training XGBoost Classifier...")
    
    # Split the data: 70% Training and 30% Testing (stratify=y to keep balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    
    # Train XGBoost Classifier
    # Using parameters to prevent overfitting: max_depth=3, subsample=0.8
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # --- Robustness Check: 5-Fold Cross Validation ---
    print("\n   -> Running 5-Fold Cross-Validation...")
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"   -> 5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model, X_train, X_test, y_train, y_test

def feature_ablation_test(X, y, top_features_to_drop):
    """
    Test model performance AFTER removing the strongest predictors.
    If accuracy stays 100% after removing top features, it suggests leakage or redundancy.
    """
    print(f"\n[Robustness Test] Training model WITHOUT top {len(top_features_to_drop)} features...")
    X_ablated = X.drop(columns=top_features_to_drop)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_ablated, y, test_size=0.30, stratify=y, random_state=42
    )
    
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, 
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, 
        eval_metric='logloss', random_state=42
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"   -> Ablated Accuracy (w/o top features): {acc:.4f}")
    return acc

def evaluate_model(model, X_test, y_test):
    """
    Step 3: Evaluation
    """
    print("\nStep 3: Evaluating Model...")
    
    y_pred = model.predict(X_test)
    
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {acc:.4f}")
    
    # Save metrics to file
    with open('metrics.txt', 'w') as f:
        f.write(f"Accuracy Score: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['Normal', 'Tumor']))
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Tumor']))
    
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Tumor'], 
                yticklabels=['Normal', 'Tumor'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion Matrix saved as 'confusion_matrix.png'")

def feature_analysis(model, X, X_train):
    """
    Step 4: Feature Analysis
    """
    print("\nStep 4: Analyzing Features...")
    
    # 1. Feature Importance from model (Gain)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    top_10_features = importance_df.head(10)['Feature'].tolist()
    
    # 2. SHAP Analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="dot", show=False, max_display=10)
    plt.title('SHAP Summary Plot (Top 10 Drivers)')
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png')
    plt.close()
    print("SHAP Summary Plot saved as 'shap_summary_plot.png'")
    
    # Top 10 Drivers horizontal bar chart (based on mean |SHAP| value) is standard in summary_plot
    # but let's make a specific bar chart for Feature Importance from XGBoost itself for clarity
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
    plt.title('Top 10 Feature Importances (XGBoost Gain)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_bar.png')
    plt.close()
    
    return top_10_features, importance_df

def visualization_suite(df_full, top_10_features):
    """
    Step 5: Visualization Suite
    """
    print("\nStep 5: Generating Visualization Suite...")
    
    # Correlation Heatmap of Top 10
    top_10_df = df_full[top_10_features] # Use the full dataframe X part
    corr_matrix = top_10_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Heatmap (Top 10 Features)')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    print("Correlation Heatmap saved as 'correlation_heatmap.png'")
    
    # Boxplots for Top 4 Features
    # We need the labels back for this
    # Re-reading or re-using isn't easy cleanly passed, so let's reconstruct X_full with Label from earlier scope or pass it
    # We will assume df_full is just X for now, we need y to plot "Tumor vs Normal"
    # To fix this, we will pass X and y combined in the main execution block
    pass 

def visualize_boxplots(X, y, top_4_features):
    data = X.copy()
    data['Label'] = y
    data['Label_Str'] = data['Label'].map({1: 'Tumor', 0: 'Normal'})
    
    plt.figure(figsize=(14, 10))
    for i, feature in enumerate(top_4_features):
        plt.subplot(2, 2, i+1)
        sns.boxplot(data=data, x='Label_Str', y=feature, palette=['#3498db', '#e74c3c'])
        plt.title(f'{feature} Distribution')
        plt.xlabel('')
        plt.ylabel('Value')
    
    plt.suptitle('Top 4 Features: Tumor vs Normal', fontsize=16)
    plt.tight_layout()
    plt.savefig('top4_boxplots.png')
    plt.close()
    print("Top 4 Boxplots saved as 'top4_boxplots.png'")

def main():
    filepath = 'merged_immune_features.csv'
    
    # 1. Data Prep
    X, y = prepare_data(filepath)
    if X is None:
        return

    # 2. Training
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # 3. Evaluation
    evaluate_model(model, X_test, y_test)
    
    # 4. Feature Analysis
    top_10_features, importance_df = feature_analysis(model, X, X_train)
    
    # 5. Visualization Suite
    # Pass X (for correlations) and top 10
    # For boxplots, we need X and y
    
    # Correlation Heatmap
    visualization_suite(X, top_10_features) # This function name was slightly separate in logic above, fixed flow here
    
    # Boxplots
    visualize_boxplots(X, y, top_10_features[:4])
    
    # Output Best Indicators
    print("\nTop 5 Best Indicators (Most Important Features):")
    for i, feature in enumerate(top_10_features[:5], 1):
        print(f"{i}. {feature}")
        
    # Run Ablation Test
    feature_ablation_test(X, y, top_10_features[:5])

if __name__ == "__main__":
    main()
