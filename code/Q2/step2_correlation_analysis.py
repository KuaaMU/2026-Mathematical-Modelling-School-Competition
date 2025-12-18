"""
Question 2: Analysis of Main Factors Affecting Water Consumption
问题2：分析影响用水量的主要因素

This script performs comprehensive analysis to identify the main factors affecting
total water consumption using:
1. Correlation analysis (Pearson and Spearman)
2. Multiple linear regression
3. Stepwise regression
4. Feature importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to support Chinese characters (if needed for output)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Create results directory
import os
os.makedirs('question2/results', exist_ok=True)

def load_data():
    """Load and preprocess data from CSV file"""
    print("="*80)
    print("STEP 1: Data Loading and Preprocessing")
    print("="*80)

    # Load data
    df = pd.read_csv('../Ａ题附件1.csv', encoding='utf-8-sig')

    # Rename columns to English for visualization
    column_mapping = {
        '年份': 'Year',
        '用水量（百亿立方米）': 'Total_Water_Consumption',
        '人口（千万人）': 'Population',
        '国内生产总值（千亿元）': 'GDP',
        '农业用水（百亿立方米）': 'Agricultural_Water',
        '工业用水（百亿立方米）': 'Industrial_Water',
        '生活用水（百亿立方米）': 'Domestic_Water',
        '生态用水（百亿立方米）': 'Ecological_Water'
    }

    df.rename(columns=column_mapping, inplace=True)

    # Use data from 2000-2016 as specified in the problem
    df_analysis = df[df['Year'] <= 2016].copy()

    print(f"\nData shape: {df_analysis.shape}")
    print(f"Time period: {df_analysis['Year'].min()} - {df_analysis['Year'].max()}")
    print("\nFirst 5 rows:")
    print(df_analysis.head())
    print("\nData statistics:")
    print(df_analysis.describe())

    return df_analysis

def correlation_analysis(df):
    """Perform Pearson and Spearman correlation analysis"""
    print("\n" + "="*80)
    print("STEP 2: Correlation Analysis")
    print("="*80)

    # Select variables for analysis
    variables = ['Total_Water_Consumption', 'Population', 'GDP',
                 'Agricultural_Water', 'Industrial_Water',
                 'Domestic_Water', 'Ecological_Water']

    df_corr = df[variables].copy()

    # Calculate Pearson correlation
    print("\nPearson Correlation Coefficients:")
    print("-" * 60)
    pearson_corr = df_corr.corr(method='pearson')
    print(pearson_corr['Total_Water_Consumption'].sort_values(ascending=False))

    # Calculate Spearman correlation
    print("\nSpearman Correlation Coefficients:")
    print("-" * 60)
    spearman_corr = df_corr.corr(method='spearman')
    print(spearman_corr['Total_Water_Consumption'].sort_values(ascending=False))

    # Visualization - Correlation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Pearson correlation heatmap
    sns.heatmap(pearson_corr, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_title('Pearson Correlation Matrix', fontsize=14, fontweight='bold')

    # Spearman correlation heatmap
    sns.heatmap(spearman_corr, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, ax=axes[1], cbar_kws={'shrink': 0.8})
    axes[1].set_title('Spearman Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('question2/results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\nCorrelation heatmap saved to: question2/results/correlation_heatmap.png")
    plt.close()

    return pearson_corr, spearman_corr

def multiple_linear_regression(df):
    """Perform multiple linear regression analysis"""
    print("\n" + "="*80)
    print("STEP 3: Multiple Linear Regression Analysis")
    print("="*80)

    # Prepare data
    X_columns = ['Population', 'GDP', 'Agricultural_Water',
                 'Industrial_Water', 'Domestic_Water', 'Ecological_Water']
    y_column = 'Total_Water_Consumption'

    X = df[X_columns].values
    y = df[y_column].values

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # Calculate metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = np.mean(np.abs(y - y_pred))

    print("\nRegression Results:")
    print("-" * 60)
    print(f"R^2 Score: {r2:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")

    print("\nRegression Coefficients:")
    print("-" * 60)
    print(f"Intercept: {model.intercept_:.6f}")
    for i, col in enumerate(X_columns):
        print(f"{col:25s}: {model.coef_[i]:10.6f}")

    # Visualization - Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(y, y_pred, alpha=0.6, s=80)
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Water Consumption', fontsize=12)
    axes[0].set_ylabel('Predicted Water Consumption', fontsize=12)
    axes[0].set_title(f'Actual vs Predicted (R^2 = {r2:.4f})', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Residual plot
    residuals = y - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=80)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Water Consumption', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('question2/results/regression_results.png', dpi=300, bbox_inches='tight')
    print("\nRegression plots saved to: question2/results/regression_results.png")
    plt.close()

    return model, r2, rmse

def stepwise_regression(df):
    """Perform stepwise regression to identify most important features"""
    print("\n" + "="*80)
    print("STEP 4: Stepwise Regression Analysis")
    print("="*80)

    X_columns = ['Population', 'GDP', 'Agricultural_Water',
                 'Industrial_Water', 'Domestic_Water', 'Ecological_Water']
    y_column = 'Total_Water_Consumption'

    X = df[X_columns].copy()
    y = df[y_column].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X_columns)

    # Forward selection
    remaining = set(X_columns)
    selected = []
    current_score = 0.0
    best_new_score = 0.0

    print("\nForward Selection Process:")
    print("-" * 60)

    while remaining:
        scores_with_candidates = []
        for candidate in remaining:
            features = selected + [candidate]
            X_train = X_scaled[features]
            model = LinearRegression()
            model.fit(X_train, y)
            y_pred = model.predict(X_train)
            score = r2_score(y, y_pred)
            scores_with_candidates.append((score, candidate))

        scores_with_candidates.sort(reverse=True)
        best_new_score, best_candidate = scores_with_candidates[0]

        if best_new_score > current_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print(f"Added: {best_candidate:25s} | R^2 = {current_score:.6f}")
        else:
            break

    print("\nFinal Selected Features:")
    print("-" * 60)
    for i, feature in enumerate(selected, 1):
        print(f"{i}. {feature}")

    # Train final model with selected features
    X_final = X_scaled[selected]
    final_model = LinearRegression()
    final_model.fit(X_final, y)
    y_pred = final_model.predict(X_final)
    final_r2 = r2_score(y, y_pred)

    print(f"\nFinal Model R^2: {final_r2:.6f}")

    # Feature importance based on standardized coefficients
    feature_importance = pd.DataFrame({
        'Feature': selected,
        'Std_Coefficient': np.abs(final_model.coef_),
        'Coefficient': final_model.coef_
    }).sort_values('Std_Coefficient', ascending=False)

    print("\nFeature Importance (Standardized Coefficients):")
    print("-" * 60)
    print(feature_importance.to_string(index=False))

    # Visualization - Feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
    bars = ax.barh(feature_importance['Feature'], feature_importance['Std_Coefficient'],
                   color=colors, edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Standardized Coefficient (Absolute Value)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, feature_importance['Std_Coefficient'])):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('question2/results/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\nFeature importance plot saved to: question2/results/feature_importance.png")
    plt.close()

    return selected, feature_importance

def analyze_factor_contributions(df):
    """Analyze the contribution of each factor to total water consumption"""
    print("\n" + "="*80)
    print("STEP 5: Factor Contribution Analysis")
    print("="*80)

    # Note: Agricultural, Industrial, Domestic, and Ecological water are COMPONENTS of total water
    component_factors = ['Agricultural_Water', 'Industrial_Water',
                        'Domestic_Water', 'Ecological_Water']

    # Calculate the proportion of each component
    proportions = df[component_factors].div(df['Total_Water_Consumption'], axis=0) * 100

    print("\nAverage Proportions of Water Consumption Components (%):")
    print("-" * 60)
    for factor in component_factors:
        mean_prop = proportions[factor].mean()
        std_prop = proportions[factor].std()
        print(f"{factor:25s}: {mean_prop:6.2f}% ± {std_prop:5.2f}%")

    # Time series visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Absolute values
    ax1 = axes[0]
    years = df['Year'].values
    ax1.plot(years, df['Total_Water_Consumption'], 'o-', linewidth=2.5,
            markersize=8, label='Total Water', color='black', zorder=5)
    ax1.plot(years, df['Agricultural_Water'], 's-', linewidth=2,
            markersize=6, label='Agricultural', alpha=0.8)
    ax1.plot(years, df['Industrial_Water'], '^-', linewidth=2,
            markersize=6, label='Industrial', alpha=0.8)
    ax1.plot(years, df['Domestic_Water'], 'D-', linewidth=2,
            markersize=6, label='Domestic', alpha=0.8)
    ax1.plot(years, df['Ecological_Water'], 'v-', linewidth=2,
            markersize=6, label='Ecological', alpha=0.8)

    ax1.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Water Consumption (10 Billion m³)', fontsize=11, fontweight='bold')
    ax1.set_title('Water Consumption by Category (Absolute Values)', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # Stacked area chart for proportions
    ax2 = axes[1]
    ax2.stackplot(years,
                 proportions['Agricultural_Water'],
                 proportions['Industrial_Water'],
                 proportions['Domestic_Water'],
                 proportions['Ecological_Water'],
                 labels=['Agricultural', 'Industrial', 'Domestic', 'Ecological'],
                 alpha=0.8)

    ax2.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Proportion (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Water Consumption Composition (Proportions)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig('question2/results/factor_contributions.png', dpi=300, bbox_inches='tight')
    print("\nFactor contribution plot saved to: question2/results/factor_contributions.png")
    plt.close()

def save_analysis_report(df, pearson_corr, spearman_corr, model, r2, rmse,
                        selected_features, feature_importance):
    """Save comprehensive analysis report"""
    print("\n" + "="*80)
    print("STEP 6: Saving Analysis Report")
    print("="*80)

    with open('question2/results/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("QUESTION 2: ANALYSIS OF MAIN FACTORS AFFECTING WATER CONSUMPTION\n")
        f.write("="*80 + "\n\n")

        f.write("1. DATA SUMMARY\n")
        f.write("-" * 60 + "\n")
        f.write(f"Time Period: {df['Year'].min()} - {df['Year'].max()}\n")
        f.write(f"Number of Observations: {len(df)}\n\n")

        f.write("2. CORRELATION ANALYSIS\n")
        f.write("-" * 60 + "\n")
        f.write("\nPearson Correlation with Total Water Consumption:\n")
        f.write(pearson_corr['Total_Water_Consumption'].sort_values(ascending=False).to_string())
        f.write("\n\nSpearman Correlation with Total Water Consumption:\n")
        f.write(spearman_corr['Total_Water_Consumption'].sort_values(ascending=False).to_string())
        f.write("\n\n")

        f.write("3. MULTIPLE LINEAR REGRESSION RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(f"R^2 Score: {r2:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n\n")

        f.write("4. STEPWISE REGRESSION - SELECTED FEATURES\n")
        f.write("-" * 60 + "\n")
        for i, feature in enumerate(selected_features, 1):
            f.write(f"{i}. {feature}\n")
        f.write("\n")

        f.write("5. FEATURE IMPORTANCE (Standardized Coefficients)\n")
        f.write("-" * 60 + "\n")
        f.write(feature_importance.to_string(index=False))
        f.write("\n\n")

        f.write("6. CONCLUSIONS\n")
        f.write("-" * 60 + "\n")
        f.write("The main factors affecting total water consumption are:\n")
        for i, row in feature_importance.iterrows():
            f.write(f"  {i+1}. {row['Feature']} (Importance: {row['Std_Coefficient']:.4f})\n")
        f.write("\n")
        f.write("Note: Total water consumption is composed of agricultural, industrial,\n")
        f.write("domestic, and ecological water usage. The analysis shows both the\n")
        f.write("compositional relationships and the influence of socio-economic factors\n")
        f.write("(Population and GDP) on overall water demand.\n")

    print("Analysis report saved to: question2/results/analysis_report.txt")

def main():
    """Main execution function"""
    print("\n")
    print("="*80)
    print(" QUESTION 2: IDENTIFYING MAIN FACTORS AFFECTING WATER CONSUMPTION ")
    print("="*80)
    print("\n")

    # Step 1: Load data
    df = load_data()

    # Step 2: Correlation analysis
    pearson_corr, spearman_corr = correlation_analysis(df)

    # Step 3: Multiple linear regression
    model, r2, rmse = multiple_linear_regression(df)

    # Step 4: Stepwise regression
    selected_features, feature_importance = stepwise_regression(df)

    # Step 5: Factor contribution analysis
    analyze_factor_contributions(df)

    # Step 6: Save report
    save_analysis_report(df, pearson_corr, spearman_corr, model, r2, rmse,
                        selected_features, feature_importance)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nAll results have been saved to the 'question2/results/' directory:")
    print("  - correlation_heatmap.png: Correlation matrices visualization")
    print("  - regression_results.png: Multiple regression analysis")
    print("  - feature_importance.png: Feature importance ranking")
    print("  - factor_contributions.png: Time series analysis of water components")
    print("  - analysis_report.txt: Comprehensive text report")
    print("\n")

if __name__ == "__main__":
    main()
