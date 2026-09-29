import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for non-technical readability
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 16
})

# 1. Generate Synthetic Dataset for Storytelling
np.random.seed(42)
n_samples = 1000

categories = ['Electronics', 'Apparel', 'Home & Kitchen', 'Beauty', 'Books']
regions = ['North America', 'Europe', 'Asia-Pacific', 'Latin America']

df = pd.DataFrame({
    'CustomerID': np.arange(1000, 1000 + n_samples),
    'Age': np.random.normal(34, 10, n_samples).astype(int),
    'Region': np.random.choice(regions, n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    'PrimaryCategory': np.random.choice(categories, n_samples, p=[0.25, 0.3, 0.2, 0.15, 0.1]),
    'Tenure_Months': np.random.randint(1, 48, n_samples),
    'Total_Spend': np.random.gamma(shape=3, scale=200, size=n_samples),
    'Discount_Responsiveness': np.random.uniform(0.1, 0.9, n_samples),
    'Satisfaction_Score': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.2, 0.45, 0.2])
})

# Introduce a structural anomaly: high discount response links to low satisfaction in Tech
df.loc[(df['PrimaryCategory'] == 'Electronics') & (df['Discount_Responsiveness'] > 0.7), 'Satisfaction_Score'] = np.random.choice([1, 2], size=len(df.loc[(df['PrimaryCategory'] == 'Electronics') & (df['Discount_Responsiveness'] > 0.7)]))

print("Dataset successfully initialized for visualization.")
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='Total_Spend', kde=True, color='#2b5c8f', bins=40)
median_spend = df['Total_Spend'].median()
plt.axvline(median_spend, color='#e056fd', linestyle='--', linewidth=2, label=f'Median Spend: ${median_spend:.2f}')
plt.title('Customer Spend Distribution: The Long-Tail Reality', pad=15)
plt.xlabel('Total Lifetime Spend ($)')
plt.ylabel('Number of Customers')
plt.legend()
plt.tight_layout()
plt.savefig('visual1_spend_distribution.png', dpi=300)
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Region', y='Total_Spend', hue='PrimaryCategory', palette='Set2')
plt.title('Where is the Value? Customer Spend Distribution by Region & Category', pad=15)
plt.xlabel('Geographic Region')
plt.ylabel('Total Spend ($)')
plt.legend(title='Product Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('visual2_regional_box.png', dpi=300)
plt.show()
g = sns.FacetGrid(df, col="PrimaryCategory", hue="Satisfaction_Score", palette="RdYlGn", col_wrap=3, height=4)
g.map(sns.scatterplot, "Discount_Responsiveness", "Total_Spend", alpha=0.7)
g.add_legend(title="Satisfaction Score")
g.set_axis_labels("Discount Sensitivity", "Total Spend ($)")
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle('The Discount Trap: Aggressive Promotions Drive Down Electronics Satisfaction', fontsize=14)
g.savefig('visual3_discount_trap.png', dpi=300)
plt.show()
plt.figure(figsize=(10, 6))
# Focus on satisfied vs unsatisfied paths
df['Loyalty_Tier'] = df['Satisfaction_Score'].apply(lambda x: 'Satisfied (4-5)' if x >= 4 else 'Unsatisfied (1-3)')
sns.lmplot(data=df, x='Tenure_Months', y='Total_Spend', hue='Loyalty_Tier', palette=['#2ed573', '#ff4757'], aspect=1.5, height=6)
plt.title('The ROI of Retention: Customer Value Trajectory Over 48 Months', pad=20)
plt.xlabel('Account Tenure (Months)')
plt.ylabel('Cumulative Revenue ($)')
plt.tight_layout()
plt.savefig('visual4_loyalty_trends.png', dpi=300)
plt.show()
plt.figure(figsize=(8, 6))
corr_matrix = df[['Age', 'Tenure_Months', 'Total_Spend', 'Discount_Responsiveness', 'Satisfaction_Score']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Correlation Strength'})
plt.title('The Metric Ecosystem: Cross-Correlations in Customer Profiles', pad=15)
plt.tight_layout()
plt.savefig('visual5_correlation_matrix.png', dpi=300)
plt.show()
