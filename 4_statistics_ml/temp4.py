"""
🎯 STATISTICS & MACHINE LEARNING CHEATSHEET
===========================================

📚 A Complete Guide to Statistical Analysis & ML Workflows

🎓 WHAT YOU'LL LEARN:
- Statistical fundamentals: descriptive stats, distributions, hypothesis testing
- End-to-end machine learning pipelines (classification & regression)
- Model evaluation, validation, and selection techniques
- Best practices for real-world data science projects

🚀 WHY THIS MATTERS:
Statistics helps you understand your data and make informed decisions.
Machine learning helps you make predictions and discover patterns.
Together, they form the foundation of data science.

📖 HOW TO USE:
Simply run this file top-to-bottom. No external files needed!
Each section builds conceptual understanding with practical examples.

💡 KEY CONCEPTS COVERED:
- Descriptive vs Inferential Statistics
- Probability Distributions & Their Uses
- Hypothesis Testing (when and why)
- Supervised Learning (Classification & Regression)
- Model Evaluation & Cross-Validation
- Overfitting vs Underfitting
- Feature Engineering & Selection

🎯 PERFECT FOR: Data science beginners, students, job interview prep
"""

# =============================================================================
# 📦 IMPORTS - Everything we need for statistical analysis and ML
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
# from pathlib import Path

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency, shapiro, normaltest

# Machine Learning - Core
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder  # , MinMaxScaler
# from sklearn.feature_selection import SelectKBest, f_classif

# ML Algorithms
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC

# ML Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    # classification_report,
    # roc_curve,
    # auc,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

# Built-in datasets for practice
from sklearn.datasets import make_classification  # , make_regression

# Model persistence
import joblib
# import pickle

# Configuration
warnings.filterwarnings("ignore")  # Clean output for learning
plt.style.use("default")
sns.set_palette("Set2")
np.random.seed(42)  # Reproducible results

print("🎉 Welcome to the Statistics & Machine Learning Cheatsheet!")
print("📊 This comprehensive guide covers statistical analysis and ML workflows")
print("🎓 Every concept is explained with WHY it matters, not just HOW to do it")
print("=" * 80)

# =============================================================================
# 🏗️ DATA GENERATION - Creating Realistic Business Dataset
# =============================================================================

print("\n🏗️ CREATING REALISTIC BUSINESS DATASET")
print("=" * 50)
print("""
💡 CONCEPT: Synthetic Data Generation
   Why create fake data? It lets us:
   - Control the relationships we want to study
   - Know the "ground truth" for testing our methods
   - Practice without privacy concerns
   - Understand different data distributions

🎲 We'll use different probability distributions to simulate real business scenarios:
   - Normal: Employee salaries (bell curve around average)
   - Gamma: Years of experience (right-skewed, most people have less experience)
   - Beta: Performance scores (bounded between 0-1, then scaled)
   - Poisson: Count data like training hours (discrete, non-negative)
""")

# Generate comprehensive business dataset
n_employees = 1200  # Good size for statistical analysis

print(f"🏭 Generating data for {n_employees} employees...")

# Create base employee data with realistic distributions
employees = pd.DataFrame(
    {
        "employee_id": range(1, n_employees + 1),
        # Categorical: Department distribution (realistic business proportions)
        "department": np.random.choice(
            ["Sales", "Engineering", "Marketing", "HR", "Finance"],
            n_employees,
            p=[0.35, 0.25, 0.20, 0.10, 0.10],  # Sales largest, HR/Finance smallest
        ),
        # Continuous: Years of experience (Gamma distribution - right skewed)
        "years_experience": np.maximum(0, np.random.gamma(2.5, 2.0, n_employees)),
        # Continuous: Base salary (Normal distribution)
        "base_salary": np.random.normal(65000, 18000, n_employees),
        # Continuous: Performance score (Beta distribution, scaled to 1-5)
        "performance_score": 1 + 4 * np.random.beta(2.5, 1.5, n_employees),
        # Count: Training hours per year (Poisson distribution)
        "training_hours": np.random.poisson(25, n_employees),
        # Binary: Remote work eligible (1=Yes, 0=No)
        "remote_eligible": np.random.binomial(1, 0.6, n_employees),
        # Discrete: Team size (Negative binomial - overdispersed count)
        "team_size": np.random.negative_binomial(3, 0.3, n_employees) + 2,
    }
)

print("""
🔗 ADDING REALISTIC CORRELATIONS
   Real business data has relationships! Let's add some:
   - Experience → Higher salary (makes sense!)
   - Training → Better performance (investment pays off)
   - Department → Salary differences (market reality)
   - Performance → More training opportunities (meritocracy)
""")

# Add realistic business correlations
employees["base_salary"] += employees["years_experience"] * 2500  # Experience premium
employees["base_salary"] += np.where(
    employees["department"] == "Engineering", 15000, 0
)  # Tech premium
employees["base_salary"] += np.where(
    employees["department"] == "Finance", 10000, 0
)  # Finance premium

# Performance correlations (with some noise to keep it realistic)
employees["performance_score"] += (
    employees["training_hours"] - 25
) * 0.02  # Training effect
employees["performance_score"] += np.random.normal(
    0, 0.3, n_employees
)  # Random variation
employees["performance_score"] = np.clip(
    employees["performance_score"], 1, 5
)  # Keep in bounds

# Ensure positive salaries
employees["base_salary"] = np.maximum(employees["base_salary"], 35000)

print(f"📊 Dataset created! Shape: {employees.shape}")
print(f"📈 First few rows:\n{employees.head()}")

# =============================================================================
# 📊 DESCRIPTIVE STATISTICS - Understanding Your Data
# =============================================================================

print("\n\n📊 DESCRIPTIVE STATISTICS - THE FOUNDATION")
print("=" * 50)
print("""
💡 CONCEPT: Descriptive vs Inferential Statistics

DESCRIPTIVE STATISTICS (What we're doing now):
- Summarize and describe the data you HAVE
- Calculate means, medians, standard deviations
- Create visualizations and identify patterns
- Answer: "What happened?"

INFERENTIAL STATISTICS (Coming later):
- Make conclusions about populations from samples
- Test hypotheses, estimate parameters
- Answer: "What can we conclude?" and "What might happen?"

🎯 WHY START WITH DESCRIPTIVE STATS?
   Before you can make predictions or test theories, you need to understand
   what your data looks like. Are there outliers? What's the typical range?
   Are variables related? This exploration guides all future analysis.
""")

# Basic descriptive statistics
numeric_columns = [
    "years_experience",
    "base_salary",
    "performance_score",
    "training_hours",
    "team_size",
]
print("📈 BASIC DESCRIPTIVE STATISTICS")
print(employees[numeric_columns].describe().round(2))

print("""
🔍 INTERPRETING THE SUMMARY TABLE:
- count: How many non-missing values (check for data quality issues)
- mean: Average value (sensitive to outliers)
- std: Standard deviation (measure of spread - higher = more varied)
- min/max: Range of values (spot potential outliers)
- 25%/50%/75%: Quartiles (understand distribution shape)

💡 QUICK DATA QUALITY CHECKS:
- Are mins/maxs reasonable? (negative salaries = problem!)
- Large difference between mean and median? (might indicate skewness)
- Very large std relative to mean? (high variability or outliers)
""")

# Detailed statistics with interpretation
print("\n🔍 DETAILED STATISTICS WITH BUSINESS INTERPRETATION")
for col in numeric_columns:
    data = employees[col]

    print(f"\n📊 {col.replace('_', ' ').title()}:")
    print(f"   📈 Mean: {data.mean():.2f}")
    print(f"   📊 Median: {data.median():.2f}")

    # Mode calculation (most frequent value, rounded for continuous data)
    rounded_data = data.round(
        0 if col in ["training_hours", "team_size"] else -3
    )  # Round appropriately
    mode_result = stats.mode(rounded_data, keepdims=True)
    print(f"   📉 Mode: {mode_result.mode[0]:.0f}")

    print(f"   📏 Standard Deviation: {data.std():.2f}")
    print(f"   📐 Variance: {data.var():.2f}")

    # Shape statistics
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)

    # Interpret skewness
    if skewness > 0.5:
        skew_interp = "Right-skewed (tail extends right, most values are lower)"
    elif skewness < -0.5:
        skew_interp = "Left-skewed (tail extends left, most values are higher)"
    else:
        skew_interp = "Approximately symmetric (balanced distribution)"

    print(f"   📊 Skewness: {skewness:.2f} - {skew_interp}")

    # Interpret kurtosis
    if kurtosis > 0:
        kurt_interp = "Heavy-tailed (more extreme values than normal distribution)"
    else:
        kurt_interp = "Light-tailed (fewer extreme values than normal distribution)"

    print(f"   📈 Kurtosis: {kurtosis:.2f} - {kurt_interp}")

    # Range statistics
    data_range = data.max() - data.min()
    iqr = data.quantile(0.75) - data.quantile(0.25)

    print(f"   📊 Range: {data_range:.2f} (max - min)")
    print(f"   🎯 IQR: {iqr:.2f} (middle 50% of data)")

    # Coefficient of variation (relative variability)
    cv = (data.std() / data.mean()) * 100
    print(
        f"   📊 Coefficient of Variation: {cv:.1f}% (std/mean - lower = more consistent)"
    )

# =============================================================================
# 🔗 CORRELATION ANALYSIS - Finding Relationships
# =============================================================================

print("\n\n🔗 CORRELATION ANALYSIS - DISCOVERING RELATIONSHIPS")
print("=" * 55)
print("""
💡 CONCEPT: Correlation vs Causation

CORRELATION measures linear relationships between variables (-1 to +1):
- +1: Perfect positive relationship (as X increases, Y increases)
-  0: No linear relationship
- -1: Perfect negative relationship (as X increases, Y decreases)

⚠️  CRITICAL: Correlation ≠ Causation!
   Just because two variables are correlated doesn't mean one causes the other.
   There might be:
   - A third variable causing both
   - Reverse causation (Y causes X, not X causes Y)
   - Pure coincidence

🎯 CORRELATION STRENGTH GUIDELINES:
   |r| < 0.3: Weak relationship
   0.3 ≤ |r| < 0.7: Moderate relationship
   |r| ≥ 0.7: Strong relationship
""")

# Calculate correlation matrix
correlation_matrix = employees[numeric_columns].corr()
print("📊 CORRELATION MATRIX:")
print(correlation_matrix.round(3))

# Find and interpret strongest correlations
correlations = []
n_vars = len(numeric_columns)
for i in range(n_vars):
    for j in range(i + 1, n_vars):
        var1, var2 = numeric_columns[i], numeric_columns[j]
        corr_val = correlation_matrix.loc[var1, var2]
        correlations.append((var1, var2, corr_val))

# Sort by absolute correlation strength
correlations.sort(key=lambda x: abs(x[2]), reverse=True)

print("\n🔥 STRONGEST CORRELATIONS (with business interpretation):")
for var1, var2, corr in correlations[:5]:  # Top 5 correlations
    # Determine strength
    abs_corr = abs(corr)
    if abs_corr >= 0.7:
        strength = "STRONG"
    elif abs_corr >= 0.3:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    direction = "POSITIVE" if corr > 0 else "NEGATIVE"

    print(f"   {var1} ↔ {var2}: {corr:.3f} ({strength} {direction})")

    # Business interpretation
    if var1 == "years_experience" and var2 == "base_salary":
        print("     💼 Makes sense: More experience typically leads to higher pay")
    elif var1 == "performance_score" and var2 == "training_hours":
        print("     📚 Logical: Training investment improves performance")
    elif "salary" in var1.lower() and "performance" in var2.lower():
        print("     ⭐ Expected: Better performers often earn more")

# Create correlation heatmap
plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Hide upper triangle
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="RdBu_r",  # Red-Blue colormap (red=negative, blue=positive)
    center=0,
    square=True,
    fmt=".3f",
    mask=mask,
    cbar_kws={"label": "Correlation Coefficient"},
)
plt.title(
    "🔗 Employee Data Correlation Matrix\n(Blue=Positive, Red=Negative, Darker=Stronger)"
)
plt.tight_layout()
plt.show()

# =============================================================================
# 🎲 PROBABILITY DISTRIBUTIONS - Understanding Data Shapes
# =============================================================================

print("\n\n🎲 PROBABILITY DISTRIBUTIONS - DATA HAS SHAPES!")
print("=" * 55)
print("""
💡 CONCEPT: Why Distribution Shape Matters

Different types of data follow different probability distributions:

📊 NORMAL DISTRIBUTION (Bell Curve):
   - Most values cluster around the mean
   - Symmetric, bell-shaped
   - Examples: Heights, test scores, measurement errors
   - Why it matters: Many statistical tests assume normality

📈 SKEWED DISTRIBUTIONS:
   - Right-skewed: Long tail to the right (income, experience)
   - Left-skewed: Long tail to the left (age at retirement)
   - Why it matters: Mean ≠ median, affects which statistics to use

🎯 PRACTICAL IMPORTANCE:
   - Choose appropriate statistical tests
   - Decide whether to transform data
   - Understand what "typical" means
   - Identify outliers appropriately
""")

# Test for normality
print("🔔 TESTING FOR NORMAL DISTRIBUTION")
print("   H0 (Null Hypothesis): Data follows normal distribution")
print("   H1 (Alternative): Data does NOT follow normal distribution")
print("   Decision rule: If p-value < 0.05, reject H0 (not normal)")

normality_results = {}
for col in ["base_salary", "years_experience", "performance_score"]:
    data = employees[col]

    # Shapiro-Wilk test (works well for smaller samples)
    if len(data) <= 5000:  # Shapiro-Wilk limit
        stat, p_value = shapiro(data)
        test_name = "Shapiro-Wilk"
    else:
        # D'Agostino's normality test (better for larger samples)
        stat, p_value = normaltest(data)
        test_name = "D'Agostino"

    is_normal = p_value > 0.05
    normality_results[col] = is_normal

    print(f"\n📊 {col}: {test_name} test")
    print(f"   Statistic: {stat:.4f}, p-value: {p_value:.4f}")
    print(
        f"   🎯 Result: {'NORMAL' if is_normal else 'NOT NORMAL'} distribution (α=0.05)"
    )

    if not is_normal:
        skewness = stats.skew(data)
        if skewness > 0.5:
            print("   💡 Suggestion: Right-skewed - consider log transformation")
        elif skewness < -0.5:
            print("   💡 Suggestion: Left-skewed - consider square transformation")

# Distribution visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("📊 Data Distribution Analysis", fontsize=16, fontweight="bold")

# Salary distribution
axes[0, 0].hist(
    employees["base_salary"], bins=40, alpha=0.7, color="skyblue", edgecolor="black"
)
axes[0, 0].axvline(
    employees["base_salary"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean: ${employees['base_salary'].mean():.0f}",
)
axes[0, 0].axvline(
    employees["base_salary"].median(),
    color="orange",
    linestyle="--",
    label=f"Median: ${employees['base_salary'].median():.0f}",
)
axes[0, 0].set_title("💰 Salary Distribution")
axes[0, 0].set_xlabel("Base Salary ($)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].legend()

# Years experience distribution
axes[0, 1].hist(
    employees["years_experience"],
    bins=30,
    alpha=0.7,
    color="lightgreen",
    edgecolor="black",
)
axes[0, 1].axvline(
    employees["years_experience"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {employees['years_experience'].mean():.1f}",
)
axes[0, 1].axvline(
    employees["years_experience"].median(),
    color="orange",
    linestyle="--",
    label=f"Median: {employees['years_experience'].median():.1f}",
)
axes[0, 1].set_title("📈 Years Experience Distribution")
axes[0, 1].set_xlabel("Years")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].legend()

# Performance score distribution
axes[1, 0].hist(
    employees["performance_score"],
    bins=25,
    alpha=0.7,
    color="lightcoral",
    edgecolor="black",
)
axes[1, 0].axvline(
    employees["performance_score"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {employees['performance_score'].mean():.2f}",
)
axes[1, 0].axvline(
    employees["performance_score"].median(),
    color="orange",
    linestyle="--",
    label=f"Median: {employees['performance_score'].median():.2f}",
)
axes[1, 0].set_title("⭐ Performance Score Distribution")
axes[1, 0].set_xlabel("Performance Score (1-5)")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].legend()

# Department distribution (categorical)
dept_counts = employees["department"].value_counts()
axes[1, 1].bar(
    dept_counts.index,
    dept_counts.values,
    alpha=0.7,
    color="lightpink",
    edgecolor="black",
)
axes[1, 1].set_title("🏢 Department Distribution")
axes[1, 1].set_xlabel("Department")
axes[1, 1].set_ylabel("Count")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# =============================================================================
# 🧪 HYPOTHESIS TESTING - Making Statistical Decisions
# =============================================================================

print("\n\n🧪 HYPOTHESIS TESTING - SCIENTIFIC DECISION MAKING")
print("=" * 55)
print("""
💡 CONCEPT: The Logic of Hypothesis Testing

Hypothesis testing is like a court trial for data:

1️⃣ NULL HYPOTHESIS (H0): "Innocent until proven guilty"
   - The status quo, what we assume is true
   - Example: "No difference between groups"

2️⃣ ALTERNATIVE HYPOTHESIS (H1): What we want to prove
   - The claim we're testing
   - Example: "There IS a difference between groups"

3️⃣ EVIDENCE: Our sample data and test statistic

4️⃣ DECISION: Based on p-value
   - p-value < 0.05: "Guilty" - reject H0, accept H1
   - p-value ≥ 0.05: "Not guilty" - fail to reject H0

⚠️  IMPORTANT: We never "accept" H0, only "fail to reject" it!

🎯 TYPES OF ERRORS:
   - Type I Error (α): Rejecting true H0 (false positive)
   - Type II Error (β): Failing to reject false H0 (false negative)
""")

# T-test: Compare salaries between departments
print("🔬 INDEPENDENT T-TEST: Engineering vs Sales Salaries")
print("""
📋 BUSINESS QUESTION: Do Engineers earn more than Sales people?
   H0: Mean Engineer salary = Mean Sales salary (no difference)
   H1: Mean Engineer salary ≠ Mean Sales salary (there is a difference)

💡 WHEN TO USE T-TEST:
   - Comparing means of two groups
   - Data is roughly normal (or large sample size)
   - Independent observations
""")

eng_salaries = employees[employees["department"] == "Engineering"]["base_salary"]
sales_salaries = employees[employees["department"] == "Sales"]["base_salary"]

print(
    f"📊 Sample sizes: Engineering (n={len(eng_salaries)}), Sales (n={len(sales_salaries)})"
)
print(f"📊 Engineering mean: ${eng_salaries.mean():.2f} ± ${eng_salaries.std():.2f}")
print(f"📊 Sales mean: ${sales_salaries.mean():.2f} ± ${sales_salaries.std():.2f}")

# Perform t-test
t_stat, p_value = stats.ttest_ind(eng_salaries, sales_salaries)
effect_size = (eng_salaries.mean() - sales_salaries.mean()) / np.sqrt(
    (
        (len(eng_salaries) - 1) * eng_salaries.var()
        + (len(sales_salaries) - 1) * sales_salaries.var()
    )
    / (len(eng_salaries) + len(sales_salaries) - 2)
)  # Cohen's d

print(f"\n📈 T-statistic: {t_stat:.4f}")
print(f"📊 P-value: {p_value:.4f}")
print(f"📊 Effect size (Cohen's d): {effect_size:.4f}")

# Interpret effect size
if abs(effect_size) < 0.2:
    effect_interp = "Small effect"
elif abs(effect_size) < 0.8:
    effect_interp = "Medium effect"
else:
    effect_interp = "Large effect"

print(f"📊 Effect size interpretation: {effect_interp}")

# Decision
if p_value < 0.05:
    print("🎯 DECISION: REJECT H0 (p < 0.05)")
    print("💡 CONCLUSION: There IS a statistically significant difference in salaries")
    higher_group = (
        "Engineers" if eng_salaries.mean() > sales_salaries.mean() else "Sales"
    )
    print(f"   📈 {higher_group} earn significantly more on average")
else:
    print("🎯 DECISION: FAIL TO REJECT H0 (p ≥ 0.05)")
    print("💡 CONCLUSION: No significant evidence of salary difference")

# Chi-square test: Department vs Performance categories
print("\n\n🔬 CHI-SQUARE TEST: Department vs Performance Level")
print("""
📋 BUSINESS QUESTION: Are certain departments more likely to have high performers?
   H0: Department and performance level are independent (no association)
   H1: Department and performance level are associated

💡 WHEN TO USE CHI-SQUARE:
   - Testing relationships between categorical variables
   - Each cell should have expected count ≥ 5
   - Independent observations
""")

# Create performance categories
performance_threshold_high = employees["performance_score"].quantile(0.75)
performance_threshold_low = employees["performance_score"].quantile(0.25)

employees["performance_category"] = pd.cut(
    employees["performance_score"],
    bins=[0, performance_threshold_low, performance_threshold_high, 5],
    labels=["Low", "Medium", "High"],
    include_lowest=True,
)

# Create contingency table
contingency_table = pd.crosstab(
    employees["department"], employees["performance_category"]
)
print("📊 CONTINGENCY TABLE (Observed frequencies):")
print(contingency_table)

# Perform chi-square test
chi2_stat, p_chi2, dof, expected_freq = chi2_contingency(contingency_table)

print(f"\n📈 Chi-square statistic: {chi2_stat:.4f}")
print(f"📊 Degrees of freedom: {dof}")
print(f"📊 P-value: {p_chi2:.4f}")
print(
    f"📊 Expected frequencies:\n{
        pd.DataFrame(
            expected_freq,
            index=contingency_table.index,
            columns=contingency_table.columns,
        ).round(1)
    }"
)

# Check assumptions
min_expected = expected_freq.min()
print(f"📊 Minimum expected frequency: {min_expected:.2f}")
if min_expected < 5:
    print("⚠️  WARNING: Chi-square assumption violated (expected frequency < 5)")
    print("   Consider combining categories or using Fisher's exact test")

# Decision
if p_chi2 < 0.05:
    print("🎯 DECISION: REJECT H0 (p < 0.05)")
    print("💡 CONCLUSION: Department and performance level ARE associated")

    # Find which departments have unusual patterns
    standardized_residuals = (contingency_table - expected_freq) / np.sqrt(
        expected_freq
    )
    print("📊 Standardized residuals (|value| > 2 indicates significant difference):")
    print(standardized_residuals.round(2))
else:
    print("🎯 DECISION: FAIL TO REJECT H0 (p ≥ 0.05)")
    print(
        "💡 CONCLUSION: No significant association between department and performance"
    )

# ANOVA: Compare performance across all departments
print("\n\n🔬 ONE-WAY ANOVA: Performance Across All Departments")
print("""
📋 BUSINESS QUESTION: Do any departments have significantly different performance?
   H0: All departments have equal mean performance scores
   H1: At least one department has different mean performance

💡 WHEN TO USE ANOVA:
   - Comparing means of 3+ groups
   - Data roughly normal within each group
   - Equal variances across groups (homoscedasticity)
   - Independent observations

🔍 ANOVA vs Multiple T-tests:
   Multiple t-tests increase Type I error rate (finding false positives)
   ANOVA controls this by testing all groups simultaneously
""")

# Group performance by department
dept_performance_groups = [
    group["performance_score"].values for name, group in employees.groupby("department")
]
dept_names = [name for name, group in employees.groupby("department")]

# Perform ANOVA
f_stat, p_anova = stats.f_oneway(*dept_performance_groups)

print("📊 DEPARTMENT PERFORMANCE MEANS:")
dept_means = employees.groupby("department")["performance_score"].agg(
    ["mean", "std", "count"]
)
print(dept_means.round(3))

print(f"\n📈 F-statistic: {f_stat:.4f}")
print(f"📊 P-value: {p_anova:.4f}")

# Test assumptions
print("\n📋 CHECKING ANOVA ASSUMPTIONS:")

# 1. Normality within groups (simplified check)
print("1️⃣ Normality: Checking largest group...")
largest_group = employees.groupby("department")["performance_score"].apply(len).idxmax()
largest_group_data = employees[employees["department"] == largest_group][
    "performance_score"
]
_, norm_p = shapiro(largest_group_data)
print(
    f"   {largest_group} normality p-value: {norm_p:.4f} ({'Normal' if norm_p > 0.05 else 'Not normal'})"
)

# 2. Equal variances (Levene's test)
levene_stat, levene_p = stats.levene(*dept_performance_groups)
print(f"2️⃣ Equal variances: Levene's test p-value: {levene_p:.4f}")
print(
    f"   {'Equal variances' if levene_p > 0.05 else 'Unequal variances (consider Welch ANOVA)'}"
)

# Decision
if p_anova < 0.05:
    print("\n🎯 DECISION: REJECT H0 (p < 0.05)")
    print(
        "💡 CONCLUSION: At least one department has significantly different performance"
    )
    print(
        "📊 POST-HOC: Would need Tukey's HSD or Bonferroni correction to find which pairs differ"
    )
else:
    print("\n🎯 DECISION: FAIL TO REJECT H0 (p ≥ 0.05)")
    print("💡 CONCLUSION: No significant difference in performance between departments")

# =============================================================================
# 🤖 MACHINE LEARNING FUNDAMENTALS
# =============================================================================

print("\n\n🤖 MACHINE LEARNING - FROM STATISTICS TO PREDICTION")
print("=" * 55)
print("""
💡 CONCEPT: Statistics vs Machine Learning

STATISTICS (what we just did):
- Understand relationships in data you HAVE
- Test hypotheses about populations
- Focus on inference and interpretation
- Example: "Is there a relationship between experience and salary?"

MACHINE LEARNING (what we're doing now):
- Predict outcomes for NEW data points
- Learn patterns from data automatically
- Focus on prediction accuracy and generalization
- Example: "Given an employee's characteristics, predict their performance"

🎯 KEY ML CONCEPTS:

SUPERVISED LEARNING: Learn from labeled examples (input → known output)
- Classification: Predict categories (spam/not spam, high/low performer)
- Regression: Predict continuous values (salary, house price, temperature)

UNSUPERVISED LEARNING: Find patterns in unlabeled data
- Clustering: Group similar data points
- Dimensionality reduction: Simplify complex data

🔍 THE ML WORKFLOW:
1. Define the problem (classification or regression?)
2. Prepare the data (cleaning, encoding, splitting)
3. Choose and train models
4. Evaluate performance
5. Select best model
6. Deploy and monitor

⚠️  CRITICAL ML CONCEPTS:
- Overfitting: Model memorizes training data, fails on new data
- Underfitting: Model too simple to capture patterns
- Bias-Variance Tradeoff: Balance between simplicity and flexibility
- Cross-validation: Robust way to estimate true performance
""")

# Prepare data for machine learning
print("🏗️ PREPARING DATA FOR MACHINE LEARNING")
print("=" * 40)

# Define classification problem: Predict high performers
print("""
🎯 CLASSIFICATION PROBLEM: Predicting High Performers

BUSINESS CASE: HR wants to identify employees likely to be high performers
to target them for retention programs, promotions, or special projects.

APPROACH: We'll predict whether an employee is a "high performer"
(top 30% by performance score) based on their other characteristics.
""")

# Create target variable for classification
performance_threshold = employees["performance_score"].quantile(0.7)  # Top 30%
employees["high_performer"] = (
    employees["performance_score"] >= performance_threshold
).astype(int)

class_distribution = employees["high_performer"].value_counts()
print(
    f"📊 Classification target: High performer (performance ≥ {performance_threshold:.2f})"
)
print(f"📊 Class distribution: {class_distribution.to_dict()}")
print(
    f"📊 Class balance: {class_distribution[1] / len(employees) * 100:.1f}% high performers"
)

if (
    class_distribution[1] / len(employees) < 0.1
    or class_distribution[1] / len(employees) > 0.9
):
    print("⚠️  WARNING: Highly imbalanced classes - may need special handling")

# Feature engineering and preparation
print("\n🔧 FEATURE ENGINEERING")
print("""
💡 CONCEPT: Feature Engineering
   Raw data rarely comes in the perfect format for ML algorithms.
   Feature engineering transforms data into formats that algorithms can use effectively.

COMMON TRANSFORMATIONS:
- Categorical → Numerical (algorithms need numbers!)
- Text → Numbers (word counts, embeddings)
- Dates → Features (day of week, month, time since)
- Scaling (make all features comparable magnitudes)
- Creating interactions (salary_per_year_experience)
""")

# Prepare features
feature_columns = [
    "years_experience",
    "base_salary",
    "training_hours",
    "team_size",
    "remote_eligible",
]

# Encode categorical variables
print("🔤 Encoding categorical variables...")
le_dept = LabelEncoder()
employees["department_encoded"] = le_dept.fit_transform(employees["department"])
feature_columns.append("department_encoded")

print(
    f"📊 Department encoding: {dict(zip(le_dept.classes_, range(len(le_dept.classes_))))}"
)

# Create feature matrix and target vector
X = employees[feature_columns].copy()
y = employees["high_performer"].copy()

print(f"📊 Feature matrix shape: {X.shape}")
print(f"📊 Features: {feature_columns}")
print("📊 Target variable: Binary (0=Normal performer, 1=High performer)")

# Data splitting
print("\n✂️ SPLITTING DATA: Training vs Testing")
print("""
💡 CONCEPT: Why Split Data?

THE PROBLEM: We want to know how well our model will work on NEW data,
but we only have the data we currently have.

THE SOLUTION: Pretend some of our data is "new" by hiding it during training.

TRAINING SET (80%): Used to train the model
- Model learns patterns from this data
- Like studying with practice problems

TEST SET (20%): Used to evaluate final performance
- Model has never seen this data during training
- Like taking the final exam

⚠️  GOLDEN RULE: Never use test data for training or model selection!
   This causes "data leakage" and overly optimistic performance estimates.
""")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # Maintain class balance in splits
)

print(
    f"📊 Training set: {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.1f}%)"
)
print(f"📊 Test set: {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.1f}%)")
print(f"📊 Training class balance: {y_train.value_counts().to_dict()}")
print(f"📊 Test class balance: {y_test.value_counts().to_dict()}")

# Feature scaling
print("\n⚖️ FEATURE SCALING")
print("""
💡 CONCEPT: Why Scale Features?

THE PROBLEM: Features have different units and scales
- Salary: $30,000 - $150,000 (large numbers)
- Performance: 1 - 5 (small numbers)
- Years experience: 0 - 30 (medium numbers)

ALGORITHMS AFFECTED: Distance-based algorithms (Logistic Regression, SVM, KNN)
treat large-scale features as more important simply because of magnitude.

SOLUTION: Transform all features to similar scales
- StandardScaler: Mean=0, Std=1 (most common)
- MinMaxScaler: Range 0-1
- RobustScaler: Uses median and IQR (better with outliers)

🔒 CRITICAL: Fit scaler on training data only, then apply to both train and test!
""")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data
X_test_scaled = scaler.transform(X_test)  # Apply same transformation to test data

print("📊 Scaling example (first 3 features):")
print("Original ranges:")
for i, col in enumerate(feature_columns[:3]):
    print(f"  {col}: {X_train.iloc[:, i].min():.2f} to {X_train.iloc[:, i].max():.2f}")

print("Scaled ranges:")
for i, col in enumerate(feature_columns[:3]):
    print(
        f"  {col}: {X_train_scaled[:, i].min():.2f} to {X_train_scaled[:, i].max():.2f}"
    )

# =============================================================================
# 🎯 CLASSIFICATION MODELS
# =============================================================================

print("\n\n🎯 CLASSIFICATION MODELS - PREDICTING CATEGORIES")
print("=" * 50)

# Model dictionary for easy iteration
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Naive Bayes": GaussianNB(),
}

# Store results
classification_results = {}

for model_name, model in models.items():
    print(f"\n🤖 {model_name.upper()}")
    print("=" * (len(model_name) + 2))

    # Model-specific explanations
    if model_name == "Logistic Regression":
        print("""
💡 LOGISTIC REGRESSION: The Linear Classifier

HOW IT WORKS:
- Uses linear combination of features to predict probability
- Applies logistic function to map any real number to 0-1 probability
- Decision boundary: Straight line (or hyperplane in higher dimensions)

STRENGTHS:
- Fast to train and predict
- Provides probability estimates
- Coefficients show feature importance and direction
- Works well when classes are roughly linearly separable

WEAKNESSES:
- Assumes linear relationship between features and log-odds
- Can struggle with complex, non-linear patterns
- Sensitive to outliers and feature scaling

WHEN TO USE: Good baseline model, when interpretability is important
""")

    elif model_name == "Decision Tree":
        print("""
💡 DECISION TREE: The Rule-Based Classifier

HOW IT WORKS:
- Creates a tree of if-else rules based on feature values
- Each split maximally separates the classes
- Predictions follow path from root to leaf

STRENGTHS:
- Highly interpretable (can visualize the rules)
- Handles both numerical and categorical features naturally
- No need for feature scaling
- Can capture non-linear patterns

WEAKNESSES:
- Prone to overfitting (especially deep trees)
- Can be unstable (small data changes → different tree)
- Biased toward features with more levels

WHEN TO USE: When interpretability is crucial, mixed data types
""")

    elif model_name == "Random Forest":
        print("""
💡 RANDOM FOREST: The Ensemble Method

HOW IT WORKS:
- Builds many decision trees with random subsets of features
- Each tree votes on the prediction
- Final prediction = majority vote (classification) or average (regression)

STRENGTHS:
- Very robust and accurate
- Handles overfitting well (averaging reduces variance)
- Provides feature importance
- Works well out-of-the-box with minimal tuning

WEAKNESSES:
- Less interpretable than single decision tree
- Can be slow with very large datasets
- Memory intensive (stores many trees)

WHEN TO USE: When accuracy is priority, good default choice
""")

    elif model_name == "Naive Bayes":
        print("""
💡 NAIVE BAYES: The Probabilistic Classifier

HOW IT WORKS:
- Uses Bayes' theorem to calculate class probabilities
- "Naive" assumption: features are independent given the class
- Assigns to class with highest posterior probability

STRENGTHS:
- Fast to train and predict
- Works well with small datasets
- Handles multiple classes naturally
- Good baseline for text classification

WEAKNESSES:
- Strong independence assumption (rarely true in practice)
- Can be outperformed by more sophisticated methods
- Requires smoothing for zero probabilities

WHEN TO USE: Text classification, small datasets, when speed is critical
""")

    # Choose appropriate features based on model
    if model_name in ["Logistic Regression", "Naive Bayes"]:
        # These models benefit from scaling
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
    else:
        # Tree-based models don't need scaling
        X_train_model = X_train
        X_test_model = X_test

    # Train model
    print("🏋️ Training model...")
    model.fit(X_train_model, y_train)

    # Make predictions
    y_pred = model.predict(X_test_model)
    y_pred_proba = model.predict_proba(X_test_model)[
        :, 1
    ]  # Probability of positive class

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("📊 PERFORMANCE METRICS:")
    print(
        f"   🎯 Accuracy:  {accuracy:.4f} ({accuracy * 100:.1f}%) - Overall correctness"
    )
    print(
        f"   🎯 Precision: {precision:.4f} ({precision * 100:.1f}%) - Of predicted high performers, how many actually are?"
    )
    print(
        f"   🎯 Recall:    {recall:.4f} ({recall * 100:.1f}%) - Of actual high performers, how many did we catch?"
    )
    print(f"   🎯 F1-Score:  {f1:.4f} - Harmonic mean of precision and recall")

    # Business interpretation
    print("\n💼 BUSINESS INTERPRETATION:")
    if precision > 0.7:
        print(
            "   ✅ High precision: When model predicts high performer, it's usually right"
        )
    else:
        print(
            "   ⚠️  Low precision: Many false positives (normal performers predicted as high)"
        )

    if recall > 0.7:
        print("   ✅ High recall: Model catches most actual high performers")
    else:
        print("   ⚠️  Low recall: Model misses many actual high performers")

    # Store results
    classification_results[model_name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predictions": y_pred,
        "probabilities": y_pred_proba,
    }

    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        print("\n🌟 FEATURE IMPORTANCE:")
        importance_df = pd.DataFrame(
            {"feature": feature_columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        for _, row in importance_df.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")

        most_important = importance_df.iloc[0]
        print(f"   💡 Most important feature: {most_important['feature']}")

    elif hasattr(model, "coef_"):
        print("\n📊 FEATURE COEFFICIENTS:")
        coef_df = pd.DataFrame(
            {"feature": feature_columns, "coefficient": model.coef_[0]}
        ).sort_values("coefficient", key=abs, ascending=False)

        for _, row in coef_df.iterrows():
            direction = "📈" if row["coefficient"] > 0 else "📉"
            print(f"   {direction} {row['feature']}: {row['coefficient']:.4f}")

# Model comparison
print("\n\n🏆 CLASSIFICATION MODEL COMPARISON")
print("=" * 35)

comparison_df = pd.DataFrame(
    {
        "Model": list(classification_results.keys()),
        "Accuracy": [
            results["accuracy"] for results in classification_results.values()
        ],
        "Precision": [
            results["precision"] for results in classification_results.values()
        ],
        "Recall": [results["recall"] for results in classification_results.values()],
        "F1-Score": [
            results["f1_score"] for results in classification_results.values()
        ],
    }
)

print(comparison_df.round(4))

# Find best model
best_model_idx = comparison_df["F1-Score"].idxmax()
best_model_name = comparison_df.loc[best_model_idx, "Model"]
best_f1 = comparison_df.loc[best_model_idx, "F1-Score"]

print(f"\n🥇 BEST MODEL: {best_model_name} (F1-Score: {best_f1:.4f})")
print("💡 F1-Score balances precision and recall - good for business decisions")

# Confusion matrices
print("\n📊 CONFUSION MATRICES")
print("Layout: [[True Negatives, False Positives], [False Negatives, True Positives]]")

for model_name, results in classification_results.items():
    cm = confusion_matrix(y_test, results["predictions"])
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{model_name}:")
    print(f"  {cm}")
    print(f"  True Negatives (correctly predicted normal): {tn}")
    print(f"  False Positives (wrongly predicted as high performer): {fp}")
    print(f"  False Negatives (missed high performers): {fn}")
    print(f"  True Positives (correctly identified high performers): {tp}")

# =============================================================================
# 📈 REGRESSION MODELS
# =============================================================================

print("\n\n📈 REGRESSION MODELS - PREDICTING CONTINUOUS VALUES")
print("=" * 55)
print("""
🎯 REGRESSION PROBLEM: Predicting Employee Salaries

BUSINESS CASE: HR wants to ensure fair compensation by predicting
what an employee's salary should be based on their characteristics.
This can help identify pay inequities and set appropriate salaries
for new hires or promotions.
""")

# Prepare regression data
regression_features = [
    "years_experience",
    "performance_score",
    "training_hours",
    "team_size",
    "remote_eligible",
    "department_encoded",
]
X_reg = employees[regression_features].copy()
y_reg = employees["base_salary"].copy()

# Split data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features for regression
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print(f"📊 Regression dataset: {X_reg.shape[0]} employees")
print(f"📊 Target statistics: Mean=${y_reg.mean():.0f}, Std=${y_reg.std():.0f}")
print(f"📊 Salary range: ${y_reg.min():.0f} - ${y_reg.max():.0f}")

# Regression models
regression_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regression": RandomForestRegressor(
        random_state=42, n_estimators=100
    ),
}

regression_results = {}

for model_name, model in regression_models.items():
    print(f"\n🎯 {model_name.upper()}")
    print("=" * (len(model_name) + 2))

    if model_name == "Linear Regression":
        print("""
💡 LINEAR REGRESSION: The Foundation of Prediction

HOW IT WORKS:
- Finds the best-fitting straight line (or hyperplane) through the data
- Minimizes sum of squared errors between predictions and actual values
- Equation: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

ASSUMPTIONS:
- Linear relationship between features and target
- Independence of observations
- Homoscedasticity (constant error variance)
- Normality of residuals

STRENGTHS:
- Simple and interpretable
- Fast to train and predict
- Coefficients show feature impact
- Well-understood statistical properties

WEAKNESSES:
- Assumes linear relationships
- Sensitive to outliers
- Can't capture complex interactions

WHEN TO USE: Baseline model, when interpretability is key, linear relationships
""")
        X_train_model = X_train_reg_scaled
        X_test_model = X_test_reg_scaled
    else:
        print("""
💡 RANDOM FOREST REGRESSION: The Flexible Predictor

HOW IT WORKS:
- Builds many decision trees with random feature subsets
- Each tree makes a prediction
- Final prediction = average of all tree predictions

STRENGTHS:
- Captures non-linear relationships
- Robust to outliers
- Handles feature interactions automatically
- Provides feature importance
- Less prone to overfitting than single decision tree

WEAKNESSES:
- Less interpretable than linear regression
- Can be slower with large datasets
- May not extrapolate well beyond training data range

WHEN TO USE: When accuracy is priority, complex relationships expected
""")
        X_train_model = X_train_reg
        X_test_model = X_test_reg

    # Train model
    print("🏋️ Training model...")
    model.fit(X_train_model, y_train_reg)

    # Make predictions
    y_pred_reg = model.predict(X_test_model)

    # Calculate regression metrics
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    mape = mean_absolute_percentage_error(y_test_reg, y_pred_reg)

    print("📊 PERFORMANCE METRICS:")
    print(f"   📊 R² Score: {r2:.4f} ({r2 * 100:.1f}% of variance explained)")
    print(f"   📊 RMSE: ${rmse:.0f} (typical prediction error)")
    print(f"   📊 MAE: ${mae:.0f} (average absolute error)")
    print(f"   📊 MAPE: {mape:.4f} ({mape * 100:.1f}% average percentage error)")

    # R² interpretation
    if r2 > 0.8:
        r2_quality = "Excellent"
    elif r2 > 0.6:
        r2_quality = "Good"
    elif r2 > 0.4:
        r2_quality = "Moderate"
    else:
        r2_quality = "Poor"

    print(f"   🎯 Model quality: {r2_quality} fit")

    # Business interpretation
    avg_salary = y_test_reg.mean()
    error_percentage = (rmse / avg_salary) * 100
    print("\n💼 BUSINESS INTERPRETATION:")
    print(f"   Average salary: ${avg_salary:.0f}")
    print(
        f"   Typical prediction error: ${rmse:.0f} ({error_percentage:.1f}% of average salary)"
    )

    if error_percentage < 10:
        print("   ✅ Very accurate predictions for HR planning")
    elif error_percentage < 20:
        print("   ✅ Reasonably accurate for most HR decisions")
    else:
        print("   ⚠️  High error rate - may need better features or different approach")

    # Store results
    regression_results[model_name] = {
        "model": model,
        "r2_score": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "predictions": y_pred_reg,
    }

    # Feature importance/coefficients
    if hasattr(model, "feature_importances_"):
        print("\n🌟 FEATURE IMPORTANCE:")
        importance_df = pd.DataFrame(
            {"feature": regression_features, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        for _, row in importance_df.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")

    elif hasattr(model, "coef_"):
        print("\n📊 FEATURE COEFFICIENTS (salary impact per unit increase):")
        coef_df = pd.DataFrame(
            {"feature": regression_features, "coefficient": model.coef_}
        ).sort_values("coefficient", key=abs, ascending=False)

        for _, row in coef_df.iterrows():
            direction = "📈" if row["coefficient"] > 0 else "📉"
            print(f"   {direction} {row['feature']}: ${row['coefficient']:.0f}")

        print(
            f"   💡 Interpretation: Each additional year of experience increases salary by ~${coef_df[coef_df['feature'] == 'years_experience']['coefficient'].iloc[0]:.0f}"
        )

# Regression model comparison
print("\n\n🏆 REGRESSION MODEL COMPARISON")
print("=" * 32)

reg_comparison_df = pd.DataFrame(
    {
        "Model": list(regression_results.keys()),
        "R² Score": [results["r2_score"] for results in regression_results.values()],
        "RMSE ($)": [results["rmse"] for results in regression_results.values()],
        "MAE ($)": [results["mae"] for results in regression_results.values()],
        "MAPE": [results["mape"] for results in regression_results.values()],
    }
)

print(reg_comparison_df.round(4))

best_reg_model_idx = reg_comparison_df["R² Score"].idxmax()
best_reg_model_name = reg_comparison_df.loc[best_reg_model_idx, "Model"]
best_r2 = reg_comparison_df.loc[best_reg_model_idx, "R² Score"]

print(f"\n🥇 BEST REGRESSION MODEL: {best_reg_model_name} (R²: {best_r2:.4f})")

# Prediction visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("🎯 Regression Model Performance: Predictions vs Actual", fontsize=14)

for idx, (model_name, results) in enumerate(regression_results.items()):
    ax = axes[idx]

    # Scatter plot of predictions vs actual
    ax.scatter(y_test_reg, results["predictions"], alpha=0.6, s=50)

    # Perfect prediction line
    min_val = min(y_test_reg.min(), results["predictions"].min())
    max_val = max(y_test_reg.max(), results["predictions"].max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Predictions"
    )

    ax.set_xlabel("Actual Salary ($)")
    ax.set_ylabel("Predicted Salary ($)")
    ax.set_title(f"{model_name}\nR² = {results['r2_score']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format axes
    ax.ticklabel_format(style="plain", axis="both")

plt.tight_layout()
plt.show()

# =============================================================================
# 🔍 MODEL VALIDATION & EVALUATION
# =============================================================================

print("\n\n🔍 ADVANCED MODEL VALIDATION")
print("=" * 35)
print("""
💡 CONCEPT: Why Cross-Validation Matters

THE PROBLEM WITH SINGLE TRAIN/TEST SPLIT:
- Performance depends on which data points end up in test set
- Might get lucky (or unlucky) with the split
- Single number doesn't show performance variability

THE SOLUTION: Cross-Validation
- Split data into K parts (folds)
- Train on K-1 parts, test on remaining part
- Repeat K times, each fold serves as test set once
- Average performance across all folds

BENEFITS:
- More reliable performance estimate
- Shows performance variability (std deviation)
- Uses all data for both training and testing
- Helps detect overfitting

🎯 COMMON CV STRATEGIES:
- K-Fold: Random splits
- Stratified K-Fold: Maintains class balance (classification)
- Time Series Split: Respects temporal order
""")

# Cross-validation for classification models
print("🤖 CROSS-VALIDATION: Classification Models")
print("Using 5-fold Stratified CV (maintains class balance)")

cv_results_class = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model_data in classification_results.items():
    model = model_data["model"]

    # Use appropriate features based on model
    if model_name in ["Logistic Regression", "Naive Bayes"]:
        X_cv = X_train_scaled
    else:
        X_cv = X_train

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_cv, y_train, cv=cv, scoring="f1")

    cv_results_class[model_name] = {
        "mean": cv_scores.mean(),
        "std": cv_scores.std(),
        "scores": cv_scores,
    }

    print(f"\n📊 {model_name}:")
    print(f"   F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   Individual folds: {[f'{score:.3f}' for score in cv_scores]}")
    print(f"   Range: {cv_scores.min():.3f} - {cv_scores.max():.3f}")

    # Stability assessment
    if cv_scores.std() < 0.05:
        print("   ✅ Very stable performance across folds")
    elif cv_scores.std() < 0.1:
        print("   ✅ Reasonably stable performance")
    else:
        print("   ⚠️  High variability - model might be sensitive to data")

# Cross-validation for regression models
print("\n🎯 CROSS-VALIDATION: Regression Models")
print("Using 5-fold CV")

cv_results_reg = {}

for model_name, model_data in regression_results.items():
    model = model_data["model"]

    # Use appropriate features
    if model_name == "Linear Regression":
        X_cv = X_train_reg_scaled
    else:
        X_cv = X_train_reg

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_cv, y_train_reg, cv=5, scoring="r2")

    cv_results_reg[model_name] = {
        "mean": cv_scores.mean(),
        "std": cv_scores.std(),
        "scores": cv_scores,
    }

    print(f"\n📊 {model_name}:")
    print(f"   R² Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   Individual folds: {[f'{score:.3f}' for score in cv_scores]}")

# Learning curves concept
print("\n📈 LEARNING CURVES CONCEPT")
print("""
💡 UNDERSTANDING MODEL BEHAVIOR

LEARNING CURVES show how performance changes with training set size:

UNDERFITTING (High Bias):
- Training and validation scores both low
- Scores converge quickly
- More data won't help much
- Need more complex model

OVERFITTING (High Variance):
- Training score high, validation score low
- Large gap between scores
- More data might help
- Need regularization or simpler model

GOOD FIT:
- Both scores high
- Small gap between scores
- Scores converge at high level
""")

# =============================================================================
# 🎛️ HYPERPARAMETER TUNING
# =============================================================================

print("\n\n🎛️ HYPERPARAMETER TUNING - OPTIMIZING PERFORMANCE")
print("=" * 55)
print("""
💡 CONCEPT: What Are Hyperparameters?

PARAMETERS vs HYPERPARAMETERS:
- Parameters: Learned by the algorithm (weights, coefficients)
- Hyperparameters: Set by you before training (learning rate, tree depth)

EXAMPLES OF HYPERPARAMETERS:
- Random Forest: n_estimators, max_depth, min_samples_split
- Logistic Regression: C (regularization strength), penalty type
- Decision Tree: max_depth, min_samples_leaf, criterion

WHY TUNE HYPERPARAMETERS?
- Default values rarely optimal for your specific dataset
- Can dramatically improve performance
- Balance between underfitting and overfitting

🔍 TUNING METHODS:
- Grid Search: Try all combinations (thorough but slow)
- Random Search: Try random combinations (faster, often as good)
- Bayesian Optimization: Smart search using past results
""")

# Grid search example with Random Forest
print("🔍 GRID SEARCH EXAMPLE: Random Forest Hyperparameters")
print("⏰ This may take a moment - we're testing many combinations...")

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],  # Number of trees
    "max_depth": [5, 10, None],  # Tree depth
    "min_samples_split": [2, 5, 10],  # Min samples to split node
    "min_samples_leaf": [1, 2, 4],  # Min samples per leaf
}

total_combinations = (
    len(param_grid["n_estimators"])
    * len(param_grid["max_depth"])
    * len(param_grid["min_samples_split"])
    * len(param_grid["min_samples_leaf"])
)

print(f"🔍 Testing {total_combinations} parameter combinations...")
print("📊 Using 3-fold CV for speed (normally would use 5-fold)")

# Perform grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,  # Faster than 5-fold for demo
    scoring="f1",
    n_jobs=-1,  # Use all CPU cores
    verbose=0,
)

grid_search.fit(X_train, y_train)

print("\n🏆 BEST HYPERPARAMETERS:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")

print("\n📊 PERFORMANCE COMPARISON:")
original_rf = RandomForestClassifier(random_state=42, n_estimators=100)
original_rf.fit(X_train, y_train)
original_pred = original_rf.predict(X_test)
original_f1 = f1_score(y_test, original_pred)

tuned_pred = grid_search.best_estimator_.predict(X_test)
tuned_f1 = f1_score(y_test, tuned_pred)

print(f"   Original Random Forest F1: {original_f1:.4f}")
print(f"   Tuned Random Forest F1: {tuned_f1:.4f}")
print(f"   Improvement: {tuned_f1 - original_f1:+.4f}")

if tuned_f1 > original_f1:
    print("   ✅ Hyperparameter tuning improved performance!")
else:
    print("   💡 No improvement - original parameters were good enough")

print(f"\n📊 BEST CV SCORE: {grid_search.best_score_:.4f}")
print("💡 CV score is typically lower than single split due to more rigorous testing")

# =============================================================================
# 💾 MODEL PERSISTENCE - SAVING AND LOADING
# =============================================================================

print("\n\n💾 MODEL PERSISTENCE - SAVING YOUR WORK")
print("=" * 45)
print("""
💡 CONCEPT: Why Save Models?

IN PRODUCTION:
- Train once, predict many times
- Models can take hours/days to train
- Consistency across different environments
- Version control for model updates

COMMON FORMATS:
- Joblib: Efficient for scikit-learn models (recommended)
- Pickle: General Python object serialization
- ONNX: Cross-platform model format
- Framework-specific: TensorFlow SavedModel, PyTorch .pt

🔒 WHAT TO SAVE:
- Trained model
- Preprocessing objects (scalers, encoders)
- Feature names and order
- Model metadata (training date, performance metrics)
""")

# Create models directory


models_dir = "saved_models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"📁 Created directory: {models_dir}")
else:
    print(f"📁 Using existing directory: {models_dir}")

# Save best models
print("\n💾 SAVING MODELS AND PREPROCESSORS")

# Best classification model
best_classifier = grid_search.best_estimator_
joblib.dump(best_classifier, f"{models_dir}/best_classifier.pkl")
print(f"✅ Saved best classifier: {best_classifier.__class__.__name__}")

# Best regression model
best_regressor = regression_results[best_reg_model_name]["model"]
joblib.dump(best_regressor, f"{models_dir}/best_regressor.pkl")
print(f"✅ Saved best regressor: {best_regressor.__class__.__name__}")

# Save preprocessing objects
joblib.dump(scaler, f"{models_dir}/classification_scaler.pkl")
joblib.dump(scaler_reg, f"{models_dir}/regression_scaler.pkl")
joblib.dump(le_dept, f"{models_dir}/department_encoder.pkl")
print("✅ Saved preprocessing objects")

# Save feature information
feature_info = {
    "classification_features": feature_columns,
    "regression_features": regression_features,
    "department_encoding": dict(zip(le_dept.classes_, range(len(le_dept.classes_)))),
    "performance_threshold": performance_threshold,
}

joblib.dump(feature_info, f"{models_dir}/feature_info.pkl")
print("✅ Saved feature information")

# Demonstrate loading and prediction
print("\n📂 LOADING MODELS AND MAKING PREDICTIONS")

# Load everything back
loaded_classifier = joblib.load(f"{models_dir}/best_classifier.pkl")
loaded_regressor = joblib.load(f"{models_dir}/best_regressor.pkl")
loaded_scaler = joblib.load(f"{models_dir}/classification_scaler.pkl")
loaded_scaler_reg = joblib.load(f"{models_dir}/regression_scaler.pkl")
loaded_encoder = joblib.load(f"{models_dir}/department_encoder.pkl")
loaded_feature_info = joblib.load(f"{models_dir}/feature_info.pkl")

print("✅ Successfully loaded all models and preprocessors")

# Example prediction for new employee
print("\n🔮 EXAMPLE: Predicting for New Employee")

new_employee_data = {
    "years_experience": 7.5,
    "base_salary": 85000,
    "training_hours": 35,
    "team_size": 6,
    "remote_eligible": 1,
    "department": "Engineering",
}

print("📊 New employee profile:")
for key, value in new_employee_data.items():
    print(f"   {key}: {value}")

# Prepare features for classification
new_emp_df = pd.DataFrame([new_employee_data])
new_emp_df["department_encoded"] = loaded_encoder.transform(
    [new_employee_data["department"]]
)[0]

# Classification prediction
class_features = new_emp_df[loaded_feature_info["classification_features"]].values
class_features_scaled = loaded_scaler.transform(class_features)
class_prediction = loaded_classifier.predict(class_features_scaled)[0]
class_probability = loaded_classifier.predict_proba(class_features_scaled)[0]

print("\n🤖 CLASSIFICATION PREDICTION:")
print(f"   High Performer: {'YES' if class_prediction == 1 else 'NO'}")
print(f"   Probability: {class_probability[1]:.3f} ({class_probability[1] * 100:.1f}%)")

# Regression prediction
reg_features = new_emp_df[loaded_feature_info["regression_features"]].values
reg_features_scaled = loaded_scaler_reg.transform(reg_features)
salary_prediction = loaded_regressor.predict(reg_features_scaled)[0]

print("\n🎯 SALARY PREDICTION:")
print(f"   Predicted salary: ${salary_prediction:,.2f}")
print(f"   Actual salary: ${new_employee_data['base_salary']:,.2f}")
print(f"   Difference: ${salary_prediction - new_employee_data['base_salary']:+,.2f}")

if abs(salary_prediction - new_employee_data["base_salary"]) < 10000:
    print("   ✅ Prediction close to actual - fair compensation")
elif salary_prediction > new_employee_data["base_salary"]:
    print("   📈 Employee may be underpaid")
else:
    print("   📉 Employee may be overpaid")

# =============================================================================
# 🎓 COMPLETE ML PIPELINE FUNCTION
# =============================================================================

print("\n\n🎓 COMPLETE ML PIPELINE - PUTTING IT ALL TOGETHER")
print("=" * 55)
print("""
💡 CONCEPT: Production-Ready ML Pipeline

A complete ML pipeline should be:
- Reproducible (same results every time)
- Modular (easy to modify components)
- Robust (handles edge cases gracefully)
- Documented (clear what each step does)
- Testable (can validate each component)

🚀 PIPELINE COMPONENTS:
1. Data loading and validation
2. Exploratory data analysis
3. Feature engineering and selection
4. Data preprocessing (scaling, encoding)
5. Model training and tuning
6. Model evaluation and selection
7. Model persistence
8. Prediction interface
""")


def complete_ml_pipeline(
    data, target_column, problem_type="classification", test_size=0.2, cv_folds=5
):
    """
    🚀 Complete Machine Learning Pipeline

    A production-ready ML pipeline that handles the entire workflow
    from raw data to trained model with proper validation.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    target_column : str
        Name of target variable column
    problem_type : str
        'classification' or 'regression'
    test_size : float
        Fraction of data for testing
    cv_folds : int
        Number of cross-validation folds

    Returns:
    --------
    dict : Pipeline results including best model, metrics, and predictions
    """

    print(f"🚀 Starting {problem_type.title()} Pipeline...")
    print(f"📊 Dataset: {data.shape[0]} samples, {data.shape[1]} features")

    # Step 1: Data preparation
    print("\n1️⃣ Data Preparation")

    # Separate features and target
    feature_cols = [col for col in data.columns if col != target_column]
    X = data[feature_cols].copy()
    y = data[target_column].copy()

    # Handle categorical variables (simple label encoding for demo)
    categorical_cols = X.select_dtypes(include=["object"]).columns
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"   Encoded {col}: {len(le.classes_)} categories")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y if problem_type == "classification" else None,
    )

    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")

    # Step 2: Preprocessing
    print("\n2️⃣ Preprocessing")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✅ Features scaled")

    # Step 3: Model selection and training
    print("\n3️⃣ Model Training & Evaluation")

    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        }
        scoring = "f1" if len(np.unique(y)) == 2 else "f1_macro"
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
        }
        scoring = "r2"

    results = {}
    best_score = -float("inf")
    best_model = None
    best_name = None

    for name, model in models.items():
        print(f"   Training {name}...")

        # Choose features based on model type
        if "Linear" in name or "Logistic" in name:
            X_train_model, X_test_model = X_train_scaled, X_test_scaled
        else:
            X_train_model, X_test_model = X_train, X_test

        # Train model
        model.fit(X_train_model, y_train)

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train_model, y_train, cv=cv_folds, scoring=scoring
        )

        # Test set prediction
        y_pred = model.predict(X_test_model)

        # Calculate metrics
        if problem_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            if len(np.unique(y)) == 2:  # Binary classification
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
                primary_metric = f1
            else:  # Multi-class
                f1_macro = f1_score(y_test, y_pred, average="macro")
                metrics = {"accuracy": accuracy, "f1_macro": f1_macro}
                primary_metric = f1_macro
        else:  # Regression
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            metrics = {"r2": r2, "rmse": rmse, "mae": mae}
            primary_metric = r2

        results[name] = {
            "model": model,
            "cv_scores": cv_scores,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_metrics": metrics,
            "test_score": primary_metric,
            "predictions": y_pred,
        }

        print(f"     CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"     Test Score: {primary_metric:.4f}")

        # Track best model
        if primary_metric > best_score:
            best_score = primary_metric
            best_model = model
            best_name = name

    print(f"\n🏆 Best Model: {best_name} (Score: {best_score:.4f})")

    # Step 4: Return comprehensive results
    pipeline_results = {
        "best_model": best_model,
        "best_model_name": best_name,
        "best_score": best_score,
        "all_results": results,
        "preprocessors": {"scaler": scaler, "encoders": encoders},
        "feature_columns": feature_cols,
        "data_splits": {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
    }

    return pipeline_results


# Demonstrate the pipeline with a new synthetic dataset
print("\n📊 PIPELINE DEMO: Creating New Dataset")

# Generate a more complex dataset
X_demo, y_demo = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=6,
    n_redundant=2,
    n_clusters_per_class=2,
    class_sep=0.8,
    random_state=42,
)

# Convert to DataFrame with realistic feature names
demo_data = pd.DataFrame(
    X_demo,
    columns=[
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
        "feature_5",
        "feature_6",
        "feature_7",
        "feature_8",
    ],
)
demo_data["target"] = y_demo

# Add a categorical feature
demo_data["category"] = np.random.choice(["A", "B", "C"], size=len(demo_data))

print(f"📊 Demo dataset created: {demo_data.shape}")

# Run the pipeline
pipeline_results = complete_ml_pipeline(
    data=demo_data,
    target_column="target",
    problem_type="classification",
    test_size=0.2,
    cv_folds=5,
)

print("\n📋 PIPELINE SUMMARY:")
print(f"   Best Model: {pipeline_results['best_model_name']}")
print(f"   Best Score: {pipeline_results['best_score']:.4f}")
print(f"   Total Models Tested: {len(pipeline_results['all_results'])}")

# =============================================================================
# 💡 ML BEST PRACTICES & COMMON PITFALLS
# =============================================================================

print("\n\n💡 MACHINE LEARNING BEST PRACTICES")
print("=" * 40)
print("""
🎯 ESSENTIAL BEST PRACTICES:

DATA QUALITY:
✅ Always explore your data first (distributions, correlations, missing values)
✅ Handle missing data appropriately (imputation vs removal)
✅ Check for and handle outliers
✅ Ensure consistent data types and formats
❌ Never assume your data is clean

FEATURE ENGINEERING:
✅ Create meaningful features from domain knowledge
✅ Scale features for distance-based algorithms
✅ Encode categorical variables properly
✅ Consider feature interactions and polynomials
❌ Don't create too many features without justification (curse of dimensionality)

MODEL SELECTION:
✅ Start with simple baselines (linear models)
✅ Try multiple algorithms and compare
✅ Use cross-validation for reliable performance estimates
✅ Consider ensemble methods for better performance
❌ Don't choose models based on single test set performance

VALIDATION:
✅ Always split data before any preprocessing
✅ Use stratified splits for classification
✅ Validate assumptions (normality, independence, etc.)
✅ Check for data leakage (future information in training)
❌ Never tune hyperparameters on the test set

INTERPRETATION:
✅ Understand what your model is actually learning
✅ Check feature importance and coefficients
✅ Validate results make business sense
✅ Consider model explainability requirements
❌ Don't deploy black-box models without understanding them
""")

print("\n⚠️ COMMON PITFALLS TO AVOID:")

pitfalls = [
    "🔴 DATA LEAKAGE: Including future information in training data",
    "🔴 OVERFITTING: Model performs great on training data, poor on new data",
    "🔴 UNDERFITTING: Model too simple to capture underlying patterns",
    "🔴 SELECTION BIAS: Non-representative samples in training/test sets",
    "🔴 SURVIVORSHIP BIAS: Only analyzing 'successful' cases",
    "🔴 CORRELATION ≠ CAUSATION: Assuming relationships imply cause and effect",
    "🔴 IGNORING CLASS IMBALANCE: When one class greatly outnumbers another",
    "🔴 NOT VALIDATING ASSUMPTIONS: Using tests without checking requirements",
    "🔴 P-HACKING: Running many tests until finding significant results",
    "🔴 EXTRAPOLATION: Predicting outside the range of training data",
]

for pitfall in pitfalls:
    print(f"   {pitfall}")

print("\n🛠️ PRACTICAL TIPS:")

tips = [
    "📊 Always plot your data before modeling",
    "🔍 Use domain expertise to guide feature engineering",
    "⚖️ Balance model complexity with interpretability needs",
    "🎯 Define success metrics aligned with business goals",
    "📈 Monitor model performance over time in production",
    "🔄 Retrain models regularly as data patterns change",
    "📚 Document your entire pipeline for reproducibility",
    "🧪 A/B test model improvements in production",
    "👥 Get feedback from domain experts and end users",
    "🎓 Keep learning - ML is a rapidly evolving field",
]

for tip in tips:
    print(f"   {tip}")

print("\n🎯 ALGORITHM SELECTION GUIDE:")

algorithm_guide = {
    "Linear/Logistic Regression": "Simple, interpretable, fast. Good baseline and when you need to understand feature impact.",
    "Decision Trees": "Highly interpretable, handles mixed data types. Use when you need explainable rules.",
    "Random Forest": "Robust, accurate, handles overfitting well. Great general-purpose algorithm.",
    "Gradient Boosting (XGBoost, LightGBM)": "Often wins competitions, handles missing values. Use for maximum accuracy.",
    "SVM": "Good with high-dimensional data, kernel trick for non-linearity. Use for text classification.",
    "Neural Networks": "Can learn complex patterns, needs lots of data. Use for images, text, complex relationships.",
    "K-Means Clustering": "Simple unsupervised learning. Use for customer segmentation, data exploration.",
    "Naive Bayes": "Fast, works well with small datasets. Use for text classification, spam detection.",
}

for algorithm, description in algorithm_guide.items():
    print(f"📊 {algorithm}:")
    print(f"   {description}")
    print()

# =============================================================================
# 🎉 CONCLUSION & NEXT STEPS
# =============================================================================

print("\n\n🎉 CONGRATULATIONS!")
print("=" * 20)
print("""
🎓 You've completed a comprehensive tour of statistics and machine learning!

📚 WHAT YOU'VE LEARNED:

STATISTICS FOUNDATIONS:
✅ Descriptive statistics (mean, median, mode, std, skewness)
✅ Correlation analysis and interpretation
✅ Probability distributions and normality testing
✅ Hypothesis testing (t-tests, chi-square, ANOVA)
✅ Confidence intervals and statistical inference

MACHINE LEARNING MASTERY:
✅ Classification vs regression problems
✅ Data preprocessing (scaling, encoding, splitting)
✅ Multiple algorithms (linear, tree-based, ensemble)
✅ Model evaluation metrics and interpretation
✅ Cross-validation and hyperparameter tuning
✅ Model persistence and deployment preparation

BEST PRACTICES:
✅ Proper data splitting and validation
✅ Avoiding overfitting and data leakage
✅ Choosing appropriate algorithms and metrics
✅ Interpreting results in business context
✅ Building reproducible pipelines

🚀 NEXT STEPS:

IMMEDIATE ACTIONS:
1. Practice with your own datasets
2. Try different algorithms on the same problem
3. Focus on feature engineering and domain knowledge
4. Build end-to-end projects to solidify learning

SKILL DEVELOPMENT:
📊 Deep dive into specific algorithms that interest you
🐍 Learn more advanced pandas and numpy techniques
📈 Study time series analysis and forecasting
🧠 Explore deep learning with TensorFlow/PyTorch
☁️ Learn cloud deployment (AWS, GCP, Azure)

PORTFOLIO PROJECTS:
🏠 Predict house prices with real estate data
📈 Build a stock price prediction model
🛒 Create a customer segmentation analysis
📝 Develop a sentiment analysis tool
🎬 Build a recommendation system

CONTINUED LEARNING:
📚 Take online courses (Coursera, edX, Udacity)
📖 Read research papers and ML blogs
🏆 Participate in Kaggle competitions
💼 Work on real business problems
👥 Join ML communities and study groups

Remember: Machine learning is as much art as science. The key is to practice,
experiment, and never stop learning!
""")

print("\n" + "=" * 80)
print("🌟 Happy Learning and May Your Models Always Converge!")
print("📧 Keep practicing, and remember: every expert was once a beginner!")
print("=" * 80)

# Final housekeeping
print("\n🧹 CLEANUP")
print("Models and results have been saved to the 'saved_models' directory")
print("You can load and use these models in future sessions")
print("This script is completely self-contained - no external files needed!")
print("\n✨ End of Statistics & Machine Learning Cheatsheet ✨")
