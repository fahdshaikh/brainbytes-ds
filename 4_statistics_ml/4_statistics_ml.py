"""
🤖 Statistics & Machine Learning Reference & Cheatsheet
Complete statistical analysis and ML pipeline examples
"""

import numpy as np
import pandas as pd
from scipy import stats

# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.datasets import load_iris  # , load_boston
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 📊 STATISTICS ESSENTIALS
# =============================================================================

print("📊 Statistical Analysis Fundamentals")
print("=" * 50)

# -- Generate Sample Dataset --
np.random.seed(42)
n_samples = 1000

# Create realistic business data
business_data = pd.DataFrame(
    {
        "employee_id": range(1, n_samples + 1),
        "department": np.random.choice(
            ["Sales", "Engineering", "Marketing", "HR"],
            n_samples,
            p=[0.4, 0.3, 0.2, 0.1],
        ),
        "years_experience": np.random.gamma(
            2, 2, n_samples
        ),  # Right-skewed distribution
        "salary": np.random.normal(65000, 15000, n_samples),
        "performance_score": np.random.beta(2, 1, n_samples) * 5,  # Scores 0-5
        "training_hours": np.random.poisson(25, n_samples),
        "remote_work_days": np.random.binomial(5, 0.6, n_samples),
    }
)

# Add some realistic correlations
business_data["salary"] += (
    business_data["years_experience"] * 2000
)  # Experience affects salary
business_data["performance_score"] += (
    business_data["training_hours"] * 0.02
)  # Training affects performance
business_data.loc[business_data["department"] == "Engineering", "salary"] *= (
    1.2  # Engineers paid more
)

print(f"📊 Business Dataset Shape: {business_data.shape}")
print(f"📈 First 5 rows:\n{business_data.head()}")

# -- Descriptive Statistics --
print("\n📈 Descriptive Statistics")
print("=" * 30)

numeric_cols = ["years_experience", "salary", "performance_score", "training_hours"]
desc_stats = business_data[numeric_cols].describe()
print(f"📊 Summary Statistics:\n{desc_stats}")

# Calculate additional statistics
for col in numeric_cols:
    data = business_data[col]
    print(f"\n📊 {col.replace('_', ' ').title()}:")
    print(f"  📈 Mean: {np.mean(data):.2f}")
    print(f"  📊 Median: {np.median(data):.2f}")

    # 🛠️ Fix for SciPy mode() change
    mode_result = stats.mode(data.round(), keepdims=True)
    mode_value = mode_result.mode.item()  # Get scalar safely
    print(f"  📉 Mode: {mode_value:.2f}")

    # print(f"  📉 Mode: {stats.mode(data.round())[0][0]:.2f}")

    print(f"  📏 Std Dev: {np.std(data):.2f}")
    print(f"  📐 Variance: {np.var(data):.2f}")
    print(f"  📊 Skewness: {stats.skew(data):.2f}")
    print(f"  📈 Kurtosis: {stats.kurtosis(data):.2f}")
    print(f"  📊 Range: {np.ptp(data):.2f}")
    print(f"  🎯 IQR: {np.percentile(data, 75) - np.percentile(data, 25):.2f}")

# -- Correlation Analysis --
print("\n🔗 Correlation Analysis")
print("=" * 25)

correlation_matrix = business_data[numeric_cols].corr()
print(f"📊 Correlation Matrix:\n{correlation_matrix}")

# Find strongest correlations
corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
        corr_value = correlation_matrix.iloc[i, j]
        corr_pairs.append((col1, col2, corr_value))

corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print("\n🔥 Strongest Correlations:")
for col1, col2, corr in corr_pairs[:3]:
    strength = (
        "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
    )
    print(f"  {col1} ↔ {col2}: {corr:.3f} ({strength})")

# =============================================================================
# 🎲 PROBABILITY & DISTRIBUTIONS
# =============================================================================

print("\n\n🎲 Probability & Distribution Analysis")
print("=" * 45)

# -- Normal Distribution Tests --
salary_data = business_data["salary"]
performance_data = business_data["performance_score"]

# Shapiro-Wilk test for normality
shapiro_salary = stats.shapiro(salary_data.sample(min(5000, len(salary_data))))
shapiro_performance = stats.shapiro(
    performance_data.sample(min(5000, len(performance_data)))
)

print("📊 Normality Tests (Shapiro-Wilk):")
print(
    f"  💰 Salary: statistic={shapiro_salary[0]:.4f}, p-value={shapiro_salary[1]:.4f}"
)
print(
    f"  ⭐ Performance: statistic={shapiro_performance[0]:.4f}, p-value={shapiro_performance[1]:.4f}"
)

# -- Distribution Fitting --
print("\n📈 Distribution Analysis:")

# Fit different distributions
distributions = [stats.norm, stats.gamma, stats.expon, stats.beta]
dist_names = ["Normal", "Gamma", "Exponential", "Beta"]

best_fits = {}
for col in ["salary", "years_experience"]:
    data = business_data[col]
    best_fit = None
    best_aic = float("inf")

    print(f"\n🔍 Analyzing {col}:")
    for dist, name in zip(distributions, dist_names):
        try:
            # Fit distribution
            params = dist.fit(data)

            # Calculate AIC (Akaike Information Criterion)
            log_likelihood = np.sum(dist.logpdf(data, *params))
            aic = 2 * len(params) - 2 * log_likelihood

            print(f"  📊 {name}: AIC = {aic:.2f}")

            if aic < best_aic:
                best_aic = aic
                best_fit = (name, params, dist)
        except Exception:
            print(f"  ❌ {name}: Could not fit")

    if best_fit:
        best_fits[col] = best_fit
        print(f"  🏆 Best fit for {col}: {best_fit[0]} (AIC: {best_aic:.2f})")

# -- Confidence Intervals --
print("\n📊 Confidence Intervals (95%):")
for col in numeric_cols:
    data = business_data[col]
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of mean
    ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=sem)
    print(f"  {col}: {mean:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")

# -- Hypothesis Testing --
print("\n🧪 Hypothesis Testing")
print("=" * 25)

# T-test: Compare salaries between departments
eng_salaries = business_data[business_data["department"] == "Engineering"]["salary"]
sales_salaries = business_data[business_data["department"] == "Sales"]["salary"]

t_stat, p_value = stats.ttest_ind(eng_salaries, sales_salaries)
print("🔬 T-test (Engineering vs Sales salaries):")
print(f"  📊 Engineering mean: ${eng_salaries.mean():.2f}")
print(f"  📊 Sales mean: ${sales_salaries.mean():.2f}")
print(f"  📈 T-statistic: {t_stat:.4f}")
print(f"  📊 P-value: {p_value:.4f}")
print(f"  🎯 Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

# Chi-square test: Department vs performance categories
business_data["performance_category"] = pd.cut(
    business_data["performance_score"],
    bins=[0, 2, 3.5, 5],
    labels=["Low", "Medium", "High"],
)

contingency_table = pd.crosstab(
    business_data["department"], business_data["performance_category"]
)
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)

print("\n🔬 Chi-square test (Department vs Performance):")
print(f"  📊 Contingency Table:\n{contingency_table}")
print(f"  📈 Chi-square statistic: {chi2:.4f}")
print(f"  📊 P-value: {p_chi2:.4f}")
print(f"  🎯 Association exists: {'Yes' if p_chi2 < 0.05 else 'No'}")

# ANOVA: Multiple group comparison
dept_groups = [
    group["salary"].values for name, group in business_data.groupby("department")
]
f_stat, p_anova = stats.f_oneway(*dept_groups)

print("\n🔬 ANOVA (Salary across all departments):")
print(f"  📈 F-statistic: {f_stat:.4f}")
print(f"  📊 P-value: {p_anova:.4f}")
print(f"  🎯 Significant differences: {'Yes' if p_anova < 0.05 else 'No'}")

# =============================================================================
# 🤖 MACHINE LEARNING - CLASSIFICATION
# =============================================================================

print("\n\n🤖 Machine Learning - Classification")
print("=" * 40)

# -- Prepare Classification Dataset --
# Predict high performance (top 30%)
threshold = business_data["performance_score"].quantile(0.7)
business_data["high_performer"] = (
    business_data["performance_score"] >= threshold
).astype(int)

print(f"🎯 Classification Target: High Performer (≥{threshold:.2f})")
print(
    f"📊 Class Distribution: {business_data['high_performer'].value_counts().to_dict()}"
)

# Prepare features
feature_cols = ["years_experience", "salary", "training_hours", "remote_work_days"]
X = business_data[feature_cols]
y = business_data["high_performer"]

# Encode categorical variables
le = LabelEncoder()
X_encoded = X.copy()
dept_encoded = le.fit_transform(business_data["department"])
X_encoded["department_encoded"] = dept_encoded
feature_cols.append("department_encoded")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Training set: {X_train.shape}, Test set: {X_test.shape}")

# -- Scale features --
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -- Logistic Regression --
print("\n📈 Logistic Regression Results:")
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print(f"  🎯 Accuracy: {lr_accuracy:.4f}")
print(f"  🎯 Precision: {lr_precision:.4f}")
print(f"  🎯 Recall: {lr_recall:.4f}")
print(f"  🎯 F1-Score: {lr_f1:.4f}")

# Feature importance
feature_importance = pd.DataFrame(
    {
        "feature": feature_cols,
        "coefficient": lr_model.coef_[0],
        "abs_coefficient": np.abs(lr_model.coef_[0]),
    }
).sort_values("abs_coefficient", ascending=False)

print("  📊 Feature Importance (Logistic Regression):")
for _, row in feature_importance.iterrows():
    print(f"    {row['feature']}: {row['coefficient']:.4f}")

# -- Random Forest Classifier --
print("\n🌲 Random Forest Results:")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"  🎯 Accuracy: {rf_accuracy:.4f}")
print(f"  🎯 Precision: {rf_precision:.4f}")
print(f"  🎯 Recall: {rf_recall:.4f}")
print(f"  🎯 F1-Score: {rf_f1:.4f}")

# Feature importance for Random Forest
rf_importance = pd.DataFrame(
    {"feature": feature_cols, "importance": rf_model.feature_importances_}
).sort_values("importance", ascending=False)

print("  📊 Feature Importance (Random Forest):")
for _, row in rf_importance.iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# -- Model Comparison --
print("\n🏆 Model Comparison:")
models_comparison = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [lr_accuracy, rf_accuracy],
        "Precision": [lr_precision, rf_precision],
        "Recall": [lr_recall, rf_recall],
        "F1-Score": [lr_f1, rf_f1],
    }
)
print(models_comparison.round(4))

# -- Confusion Matrix --
print("\n📊 Confusion Matrices:")
lr_cm = confusion_matrix(y_test, lr_pred)
rf_cm = confusion_matrix(y_test, rf_pred)

print("  Logistic Regression:")
print(f"    {lr_cm}")
print("  Random Forest:")
print(f"    {rf_cm}")

# =============================================================================
# 🎯 MACHINE LEARNING - REGRESSION
# =============================================================================

print("\n\n🎯 Machine Learning - Regression")
print("=" * 35)

# -- Prepare Regression Dataset --
# Predict salary based on other features
feature_cols_reg = [
    "years_experience",
    "performance_score",
    "training_hours",
    "remote_work_days",
]
X_reg = business_data[feature_cols_reg]
y_reg = business_data["salary"]

# Add department encoding
X_reg_encoded = X_reg.copy()
X_reg_encoded["department_encoded"] = dept_encoded

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_encoded, y_reg, test_size=0.2, random_state=42
)

# Scale features
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

print(f"📊 Regression Dataset: {X_reg_encoded.shape}")
print(f"🎯 Target: Salary (${y_reg.mean():.2f} ± ${y_reg.std():.2f})")

# -- Linear Regression --
print("\n📈 Linear Regression Results:")
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg_scaled, y_train_reg)

lin_pred = lin_reg.predict(X_test_reg_scaled)

# Calculate regression metrics
lin_mse = mean_squared_error(y_test_reg, lin_pred)
lin_mae = mean_absolute_error(y_test_reg, lin_pred)
lin_r2 = r2_score(y_test_reg, lin_pred)
lin_rmse = np.sqrt(lin_mse)

print(f"  📊 R² Score: {lin_r2:.4f}")
print(f"  📊 RMSE: ${lin_rmse:.2f}")
print(f"  📊 MAE: ${lin_mae:.2f}")
print(f"  📊 MSE: {lin_mse:.2f}")

# Coefficients
lin_coeffs = pd.DataFrame(
    {"feature": list(X_reg_encoded.columns), "coefficient": lin_reg.coef_}
).sort_values("coefficient", key=abs, ascending=False)

print("  📊 Coefficients:")
for _, row in lin_coeffs.iterrows():
    print(f"    {row['feature']}: ${row['coefficient']:.2f}")

# -- Random Forest Regression --
print("\n🌲 Random Forest Regression Results:")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)

rf_pred_reg = rf_reg.predict(X_test_reg)

rf_mse = mean_squared_error(y_test_reg, rf_pred_reg)
rf_mae = mean_absolute_error(y_test_reg, rf_pred_reg)
rf_r2 = r2_score(y_test_reg, rf_pred_reg)
rf_rmse = np.sqrt(rf_mse)

print(f"  📊 R² Score: {rf_r2:.4f}")
print(f"  📊 RMSE: ${rf_rmse:.2f}")
print(f"  📊 MAE: ${rf_mae:.2f}")
print(f"  📊 MSE: {rf_mse:.2f}")

# Feature importance
rf_reg_importance = pd.DataFrame(
    {"feature": list(X_reg_encoded.columns), "importance": rf_reg.feature_importances_}
).sort_values("importance", ascending=False)

print("  📊 Feature Importance:")
for _, row in rf_reg_importance.iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# -- Regression Model Comparison --
print("\n🏆 Regression Model Comparison:")
reg_comparison = pd.DataFrame(
    {
        "Model": ["Linear Regression", "Random Forest"],
        "R² Score": [lin_r2, rf_r2],
        "RMSE": [lin_rmse, rf_rmse],
        "MAE": [lin_mae, rf_mae],
    }
)
print(reg_comparison.round(4))

# =============================================================================
# 🔍 MODEL EVALUATION & VALIDATION
# =============================================================================

print("\n\n🔍 Advanced Model Evaluation")
print("=" * 35)

# -- Cross-Validation --
print("🔄 Cross-Validation Results:")

# Classification CV
lr_cv_scores = cross_val_score(
    lr_model, X_train_scaled, y_train, cv=5, scoring="accuracy"
)
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="accuracy")

print(
    f"  📊 Logistic Regression CV: {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}"
)
print(f"  📊 Random Forest CV: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")

# Regression CV
lin_cv_scores = cross_val_score(
    lin_reg, X_train_reg_scaled, y_train_reg, cv=5, scoring="r2"
)
rf_reg_cv_scores = cross_val_score(rf_reg, X_train_reg, y_train_reg, cv=5, scoring="r2")

print(
    f"  📊 Linear Regression CV R²: {lin_cv_scores.mean():.4f} ± {lin_cv_scores.std():.4f}"
)
print(
    f"  📊 RF Regression CV R²: {rf_reg_cv_scores.mean():.4f} ± {rf_reg_cv_scores.std():.4f}"
)

# -- Hyperparameter Tuning --
print("\n⚙️ Hyperparameter Tuning (Random Forest):")

# Grid search for Random Forest Classifier
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="f1", n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"  🏆 Best Parameters: {grid_search.best_params_}")
print(f"  📊 Best CV Score: {grid_search.best_score_:.4f}")

# Test the best model
best_rf = grid_search.best_estimator_
best_rf_pred = best_rf.predict(X_test)
best_rf_accuracy = accuracy_score(y_test, best_rf_pred)
best_rf_f1 = f1_score(y_test, best_rf_pred)

print(f"  🎯 Test Accuracy: {best_rf_accuracy:.4f}")
print(f"  🎯 Test F1-Score: {best_rf_f1:.4f}")

# =============================================================================
# 🏗️ REAL-WORLD ML PIPELINE EXAMPLE
# =============================================================================

print("\n\n🏗️ Complete ML Pipeline Example")
print("=" * 35)

# Load famous datasets for demonstration
print("📊 Using Iris Dataset for Classification Pipeline:")

# Load Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["species"] = iris.target
iris_df["species_name"] = iris_df["species"].map(
    {0: "setosa", 1: "versicolor", 2: "virginica"}
)

print(f"🌸 Iris Dataset Shape: {iris_df.shape}")
print(f"📊 Species Distribution: {iris_df['species_name'].value_counts().to_dict()}")


# Complete pipeline
def ml_pipeline(X, y, test_size=0.2, random_state=42):
    """🚀 Complete ML Pipeline"""

    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train multiple models
    models = {
        "Logistic Regression": LogisticRegression(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
    }

    results = {}

    for name, model in models.items():
        # Fit model
        if "Logistic" in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Cross-validation
        if "Logistic" in name:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "predictions": y_pred,
            "probabilities": y_pred_proba,
        }

    return results, X_test, y_test, scaler


# Run pipeline on Iris dataset
X_iris = iris_df[iris.feature_names]
y_iris = iris_df["species"]

pipeline_results, X_test_iris, y_test_iris, iris_scaler = ml_pipeline(X_iris, y_iris)

# Display results
print("\n🏆 Pipeline Results Summary:")
results_df = pd.DataFrame(
    {
        model_name: {
            "Accuracy": results["accuracy"],
            "Precision": results["precision"],
            "Recall": results["recall"],
            "F1-Score": results["f1_score"],
            "CV Mean": results["cv_mean"],
            "CV Std": results["cv_std"],
        }
        for model_name, results in pipeline_results.items()
    }
).T

print(results_df.round(4))

# Detailed classification report for best model
best_model_name = results_df["Accuracy"].idxmax()
best_results = pipeline_results[best_model_name]

print(f"\n🥇 Best Model: {best_model_name}")
print("📊 Detailed Classification Report:")
print(
    classification_report(
        y_test_iris, best_results["predictions"], target_names=iris.target_names
    )
)

# Confusion matrix
cm = confusion_matrix(y_test_iris, best_results["predictions"])
print("\n📊 Confusion Matrix:")
print(cm)

# =============================================================================
# 💡 PRACTICAL ML INSIGHTS & TIPS
# =============================================================================

print("\n\n💡 ML Best Practices & Insights")
print("=" * 35)

insights = [
    "🎯 **Feature Engineering**: Create meaningful features from raw data",
    "📊 **Data Quality**: Clean data is more important than complex algorithms",
    "🔄 **Cross-Validation**: Always validate with unseen data",
    "⚖️ **Bias-Variance**: Balance model complexity with generalization",
    "📈 **Metrics Matter**: Choose metrics that align with business goals",
    "🔍 **Interpretability**: Understand what your model is learning",
    "🚀 **Start Simple**: Begin with simple models before complex ones",
    "📚 **Domain Knowledge**: Incorporate subject matter expertise",
]

for insight in insights:
    print(f"  {insight}")

# Performance tips
print("\n⚡ Performance Optimization Tips:")
perf_tips = [
    "Use vectorized operations with NumPy/Pandas",
    "Consider feature selection for high-dimensional data",
    "Use appropriate scaling for different algorithms",
    "Implement early stopping for iterative algorithms",
    "Cache results for expensive computations",
    "Use parallel processing where possible",
]

for i, tip in enumerate(perf_tips, 1):
    print(f"  {i}. {tip}")

# Model selection guidelines
print("\n🎯 Model Selection Guidelines:")
model_guide = {
    "Linear/Logistic Regression": "Fast, interpretable, good baseline",
    "Random Forest": "Robust, handles mixed data types, feature importance",
    "SVM": "Good for high-dimensional data, kernel trick",
    "Neural Networks": "Complex patterns, large datasets, needs tuning",
    "Gradient Boosting": "High performance, handles missing values",
    "Naive Bayes": "Fast, good for text classification, small datasets",
}

for model, description in model_guide.items():
    print(f"  📊 {model}: {description}")

print("\n🎉 Statistics & ML Analysis Complete!")
print("📚 Ready to apply these techniques to real-world problems!")


# -- Sample Prediction Function --
def make_prediction_example():
    """🔮 Example prediction function"""
    # Using the best model from our pipeline
    best_model = pipeline_results[best_model_name]["model"]

    # Example: Predict species for new iris measurements
    new_sample = [[5.1, 3.5, 1.4, 0.2]]  # New iris measurements

    if "Logistic" in best_model_name:
        new_sample_scaled = iris_scaler.transform(new_sample)
        prediction = best_model.predict(new_sample_scaled)[0]
        probabilities = best_model.predict_proba(new_sample_scaled)[0]
    else:
        prediction = best_model.predict(new_sample)[0]
        probabilities = best_model.predict_proba(new_sample)[0]

    species_name = iris.target_names[prediction]

    print("\n🔮 Prediction Example:")
    print(f"  📊 Input: {new_sample[0]}")
    print(f"  🎯 Predicted Species: {species_name}")
    print(f"  📊 Probabilities: {dict(zip(iris.target_names, probabilities.round(3)))}")


make_prediction_example()
