"""
📊 Statistics & Machine Learning – From "What's This Data?" to "I Can Predict Things!"
=====================================================================================

Hey there! Ready to go from staring at spreadsheets confused to actually making sense of data?
This is your hands-on guide to statistics and machine learning – no PhD required, just curiosity and Python.

🎯 What you'll actually learn:
- How to describe your data without boring everyone to death
- When your results are "statistically significant" vs just lucky
- Building models that predict stuff (and knowing when they're lying to you)
- The difference between "my model works!" and "my model works in the real world"

🤔 Why this matters:
Ever wondered if that A/B test result is real or just random noise? Want to predict customer churn without just guessing?
Statistics tells you what's actually happening in your data. ML helps you make predictions.
Together, they're like having superpowers for data.

🚀 How to use this:
Just run the whole file top to bottom – it's designed to work without any external data files.
I've included tons of examples with a fake employee dataset, so you can see everything in action.
Each section builds on the last, but feel free to jump around if something catches your eye.

💡 What's inside:
- Statistics basics (mean, median – yeah, but also the interesting stuff)
- Hypothesis testing (is this difference real or am I imagining it?)
- Regression (predicting numbers) and classification (predicting categories)
- How to know if your model is actually good or just memorizing
- Common mistakes that'll make you look silly (so you can avoid them)

🎓 Perfect if you're:
Learning data science, prepping for interviews, or just tired of not knowing what "p-value" means.

📦 Requirements:
The usual suspects: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy
Most of these come with Anaconda, or just pip install them.

🤖 Note:
This is part of the [Cheatsheet DS Minicourse] – a project built with curiosity, coffee,
and some help from Claude AI to make learning data science less painful.

Let's turn that data into insights! 📈
"""

# -----------------------------------------------------------------------------
# IMPORTS - All libraries needed for this cheatsheet
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)  # , mean_squared_error, r2_score)
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

print("Statistics & Machine Learning Cheatsheet")
print("-" * 50)


# -----------------------------------------------------------------------------
# 📊 SAMPLE DATASETS - Creating realistic fake data for examples
# -----------------------------------------------------------------------------

# WHY WE CREATE FAKE DATA:
# Instead of hunting for "the perfect dataset," we're creating one that has realistic
# relationships - like how salary typically increases with experience and education.
# This way, you can see exactly how each technique works without getting distracted
# by messy real-world data quirks.

# Create a sample dataset for demonstrations
np.random.seed(42)
n_samples = 200

# Employee dataset for regression examples
employees_data = {
    "age": np.random.normal(35, 8, n_samples).astype(int),
    "years_experience": np.random.normal(8, 4, n_samples),
    "education_level": np.random.choice(
        [1, 2, 3, 4], n_samples, p=[0.2, 0.3, 0.3, 0.2]
    ),
    "department": np.random.choice(
        ["Engineering", "Sales", "Marketing", "HR"], n_samples
    ),
    "performance_score": np.random.normal(75, 12, n_samples),
}

# Create salary based on other factors (realistic relationship)
employees_data["salary"] = (
    30000
    + employees_data["years_experience"] * 3500
    + employees_data["education_level"] * 8000
    + employees_data["performance_score"] * 200
    + np.random.normal(0, 5000, n_samples)
)

employees_df = pd.DataFrame(employees_data)
employees_df["high_performer"] = (employees_df["performance_score"] > 80).astype(int)

print("Sample dataset created with", len(employees_df), "employee records")
print("\nDataset preview:")
print(employees_df.head())


# -----------------------------------------------------------------------------
# 📈 DESCRIPTIVE STATISTICS - Understanding Your Data
# -----------------------------------------------------------------------------

# WHAT IS DESCRIPTIVE STATISTICS?
# It's your "data detective" toolkit - you use it to understand what your data looks like
# BEFORE you start making fancy predictions. Think of it as taking a good look around
# a new city before you start navigating.
#
# WHY START HERE?
# Because you can't build good models on data you don't understand. If you jump straight
# to machine learning without knowing your data, you'll probably build something that
# looks impressive but doesn't actually work.

print("\n" + "-" * 60)
print("DESCRIPTIVE STATISTICS")
print("-" * 60)

# CENTRAL TENDENCY - "What's normal in this dataset?"
# These three numbers tell you where most of your data hangs out
salaries = employees_df["salary"]
ages = employees_df["age"]

print("Central Tendency Measures:")
print(
    f"Mean salary: ${salaries.mean():,.2f}"
)  # Average - gets pulled by extreme values
print(
    f"Median salary: ${salaries.median():,.2f}"
)  # Middle value - not affected by outliers
print(f"Mode age: {ages.mode().iloc[0]} years")  # Most common value

# WHY ALL THREE?
# Mean tells you the mathematical average, but if you have a few super-high earners,
# it might not represent "typical." Median is the middle person - often more realistic.
# Mode shows what's most common. In our employee data, if mean > median, we probably
# have some high earners pulling the average up.

# MEASURES OF SPREAD - "How scattered is this data?"
# These tell you if everyone earns roughly the same, or if there's huge variation
print("\nMeasures of Spread:")
print(
    f"Salary standard deviation: ${salaries.std():,.2f}"
)  # How spread out salaries are
print(f"Salary variance: ${salaries.var():,.0f}")  # Standard deviation squared
print(f"Age range: {ages.min()} - {ages.max()} years")  # Simplest measure of spread

# INTERPRETING STANDARD DEVIATION:
# If std dev is small relative to the mean, most people earn similar amounts.
# If it's large, there's a big salary gap in your company.
# Rule of thumb: ~68% of values fall within 1 std dev of the mean in normal distributions.

# PERCENTILES - "Where do I rank?"
# These are super useful for understanding your position relative to others
print("\nPercentiles (salary):")
for p in [25, 50, 75, 90]:
    print(f"{p}th percentile: ${np.percentile(salaries, p):,.2f}")

# WHAT PERCENTILES MEAN:
# 25th percentile = 25% of people earn less than this amount
# 75th percentile = you're in the top 25% if you earn more than this
# The gap between 75th and 25th (called IQR) shows the "middle 50%" range

# DISTRIBUTION SHAPE - "Is this data normal or weird?"
print("\nDistribution Analysis:")
print(f"Salary skewness: {salaries.skew():.3f}")  # 0 = normal, >0 = right skew
print(f"Salary kurtosis: {salaries.kurtosis():.3f}")  # 0 = normal, >0 = heavy tails

# SKEWNESS INTERPRETATION:
# Skewness > 0: A few high earners are pulling the distribution right (common in salaries)
# Skewness < 0: A few low values pulling left
# Skewness ≈ 0: Pretty symmetric, like a bell curve
#
# WHY CARE? Many statistical tests assume normal distribution. If your data is highly
# skewed, you might need to transform it or use different tests.


# -----------------------------------------------------------------------------
# 📊 INFERENTIAL STATISTICS - Making Conclusions from Samples
# -----------------------------------------------------------------------------

# WHAT IS INFERENTIAL STATISTICS?
# You have data from 200 employees, but you want to make statements about ALL employees
# at similar companies. That's the leap from "what I observed" to "what I can conclude."
# It's like tasting one spoonful of soup to judge the whole pot.
#
# THE BIG QUESTION: How confident can I be that what I see in my sample reflects reality?

print("\n" + "-" * 60)
print("INFERENTIAL STATISTICS")
print("-" * 60)

# CONFIDENCE INTERVALS - "I'm 95% sure the true average is between X and Y"
# This is like saying: "Based on my sample, I'm pretty confident the real population
# average salary is somewhere in this range"
confidence_level = 0.95
alpha = 1 - confidence_level
sample_mean = salaries.mean()
sample_std = salaries.std()
sample_size = len(salaries)

# Calculate 95% confidence interval for mean salary
margin_of_error = stats.t.ppf(1 - alpha / 2, sample_size - 1) * (
    sample_std / np.sqrt(sample_size)
)
ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print("95% Confidence Interval for mean salary:")
print(f"${ci_lower:,.2f} to ${ci_upper:,.2f}")

# WHAT THIS ACTUALLY MEANS:
# If we repeated this study 100 times with different random samples of 200 employees,
# about 95 of those times, the true population mean would fall within our calculated range.
# It's NOT saying "there's a 95% chance the true mean is in this range" - that's a
# common misunderstanding!

# CENTRAL LIMIT THEOREM - "Why sample averages are magical"
# This is one of the most important concepts in statistics, and here's why:
# Even if your original data is weirdly distributed, sample averages will be normally
# distributed if your sample size is big enough (usually n > 30).
print("\nCentral Limit Theorem Demo:")
print(f"Population mean: ${sample_mean:,.2f}")
print(
    f"Standard error: ${sample_std / np.sqrt(sample_size):,.2f}"
)  # How precise our estimate is

# Let's prove it by taking multiple samples
sample_means = []
for _ in range(100):
    sample_30 = np.random.choice(salaries, 30, replace=True)  # Take 30 random employees
    sample_means.append(sample_30.mean())

print(f"Mean of 100 sample means: ${np.mean(sample_means):,.2f}")
print(f"Standard deviation of sample means: ${np.std(sample_means):,.2f}")

# MAGIC HAPPENING HERE:
# Notice how the mean of sample means ≈ population mean, and the standard deviation
# of sample means ≈ standard error we calculated above. This is CLT in action!
# This is why we can make inferences about populations from samples.


# -----------------------------------------------------------------------------
# 🧪 HYPOTHESIS TESTING - Statistical Significance Testing
# -----------------------------------------------------------------------------

# WHAT IS HYPOTHESIS TESTING?
# It's the "is this difference real or just random luck?" toolkit.
# Think of it like a court trial: you start by assuming innocence (null hypothesis),
# then see if the evidence is strong enough to prove guilt (reject null).
#
# THE PROCESS:
# 1. Set up null hypothesis (H0): "nothing interesting is happening"
# 2. Set up alternative hypothesis (H1): "something IS happening"
# 3. Calculate how "weird" your data would be if H0 were true
# 4. If it's weird enough (p < 0.05), maybe H0 is wrong
#
# REAL WORLD EXAMPLE: You flip a coin 10 times and get 8 heads.
# H0: Coin is fair  H1: Coin is biased
# Question: How often would a fair coin give 8+ heads in 10 flips?

print("\n" + "-" * 60)
print("HYPOTHESIS TESTING")
print("-" * 60)

# ONE-SAMPLE T-TEST - "Is our average different from a claimed value?"
# Scenario: Someone claims the average salary in your industry is $65,000.
# You have data from 200 employees. Is their claim accurate?
print("One-Sample T-Test:")
print("H0: Mean salary = $65,000 (null hypothesis)")
print("H1: Mean salary ≠ $65,000 (alternative hypothesis)")

claimed_mean = 65000
t_stat, p_value = stats.ttest_1samp(salaries, claimed_mean)

print(f"T-statistic: {t_stat:.3f}")  # How many standard errors away from claimed mean
print(f"P-value: {p_value:.6f}")  # Probability of seeing this difference by chance
print(f"Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} (α = 0.05)")

# INTERPRETING THE RESULTS:
# T-statistic: How far our sample mean is from $65,000 in "standard error units"
# P-value: If the true mean really WAS $65,000, what's the chance we'd see a
#          difference this big just from random sampling?
# If p < 0.05: Either we got very unlucky, or the claimed mean is wrong

# TWO-SAMPLE T-TEST - "Are these two groups actually different?"
# Real scenario: Do Engineers earn more than Sales people, or is any difference
# we see just due to random sampling?
eng_salaries = employees_df[employees_df["department"] == "Engineering"]["salary"]
sales_salaries = employees_df[employees_df["department"] == "Sales"]["salary"]

print("\nTwo-Sample T-Test (Engineering vs Sales):")
print("H0: No difference in mean salaries")
print(f"Engineering mean: ${eng_salaries.mean():,.2f}")
print(f"Sales mean: ${sales_salaries.mean():,.2f}")

t_stat2, p_value2 = stats.ttest_ind(eng_salaries, sales_salaries)
print(f"P-value: {p_value2:.6f}")
print(
    f"Result: {'Significant difference' if p_value2 < 0.05 else 'No significant difference'}"
)

# WHY TWO-SAMPLE T-TEST?
# When you want to compare means between two independent groups.
# The test accounts for the fact that both groups have sampling variability.
# If p < 0.05, the salary difference is probably real, not just random.

# CHI-SQUARE TEST - "Are these categorical variables related?"
# Question: Is being a high performer related to which department you work in?
# Or is it just random who ends up being high performers?
print("\nChi-Square Test of Independence:")
print("Testing if department and high performance are independent")

contingency_table = pd.crosstab(
    employees_df["department"], employees_df["high_performer"]
)
chi2, p_val_chi, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-square statistic: {chi2:.3f}")
print(f"P-value: {p_val_chi:.6f}")
print(
    f"Result: {'Variables are dependent' if p_val_chi < 0.05 else 'Variables are independent'}"
)

# WHEN TO USE CHI-SQUARE:
# When both variables are categorical (department, gender, yes/no responses).
# It compares what you observed vs what you'd expect if there was no relationship.
# If p < 0.05: There's probably a real association between the variables.

# ANOVA - "Are there differences across multiple groups?"
# Question: Do people with different education levels (1,2,3,4) have different salaries?
# t-test only works for 2 groups, ANOVA works for 2+ groups
print("\nOne-Way ANOVA (salary by education level):")
groups = []
for level in [1, 2, 3, 4]:
    groups.append(employees_df[employees_df["education_level"] == level]["salary"])

f_stat, p_val_anova = stats.f_oneway(*groups)
print(f"F-statistic: {f_stat:.3f}")
print(f"P-value: {p_val_anova:.6f}")
print(
    f"Result: {'Significant differences exist' if p_val_anova < 0.05 else 'No significant differences'}"
)

# ANOVA INTERPRETATION:
# F-statistic: Ratio of between-group variance to within-group variance
# If groups have very different means, F will be large
# If p < 0.05: At least one group is significantly different from the others
# (But ANOVA doesn't tell you WHICH groups are different - need post-hoc tests for that)


# -----------------------------------------------------------------------------
# 📈 REGRESSION MODELS - Predicting Continuous Values
# -----------------------------------------------------------------------------

# WHAT IS REGRESSION?
# It's about finding relationships between variables to make predictions.
# Like: "If I know someone's experience and education, can I predict their salary?"
#
# THE GOAL: Build a mathematical equation that maps inputs (features) to outputs (targets)
# Real world uses: Predicting house prices, stock values, customer lifetime value, etc.

print("\n" + "-" * 60)
print("REGRESSION MODELS")
print("-" * 60)

# SIMPLE LINEAR REGRESSION - "One input predicts one output"
# Question: Can years of experience alone predict salary?
# We're fitting a straight line: salary = intercept + (slope × experience)
print("Simple Linear Regression (Experience → Salary):")
X_simple = employees_df[["years_experience"]]  # Must be 2D for sklearn
y = employees_df["salary"]

model_simple = LinearRegression()
model_simple.fit(X_simple, y)

print(f"Coefficient: ${model_simple.coef_[0]:,.2f} per year of experience")
print(f"Intercept: ${model_simple.intercept_:,.2f}")  # Salary with 0 years experience
print(f"R-squared: {model_simple.score(X_simple, y):.3f}")

# INTERPRETING THE RESULTS:
# Coefficient: For each additional year of experience, salary increases by $X
# Intercept: Expected salary for someone with 0 years experience
# R-squared: % of salary variation explained by experience (higher = better fit)
# R² of 0.3 means experience explains 30% of salary differences

# MULTIPLE LINEAR REGRESSION - "Multiple inputs for better predictions"
# Now we use experience + education + performance to predict salary
# Formula: salary = b0 + (b1×experience) + (b2×education) + (b3×performance)
print("\nMultiple Linear Regression:")
X_multiple = employees_df[["years_experience", "education_level", "performance_score"]]

model_multiple = LinearRegression()
model_multiple.fit(X_multiple, y)

feature_names = ["years_experience", "education_level", "performance_score"]
print("Coefficients:")
for name, coef in zip(feature_names, model_multiple.coef_):
    print(f"  {name}: {coef:,.2f}")

print(f"R-squared: {model_multiple.score(X_multiple, y):.3f}")

# WHY MULTIPLE REGRESSION?
# Single variables rarely explain everything. Using multiple predictors usually gives
# better predictions. Each coefficient tells you the effect of that variable while
# holding all others constant. Notice how R² improved from simple to multiple!

# POLYNOMIAL REGRESSION - "When straight lines aren't enough"
# Sometimes the relationship isn't linear. Maybe salary increases slowly at first,
# then rapidly with experience. Polynomial features can capture curves.
print("\nPolynomial Regression (degree 2):")
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_simple)  # Creates x, x², x×x interactions

model_poly = LinearRegression()
model_poly.fit(X_poly, y)
print(f"Polynomial R-squared: {model_poly.score(X_poly, y):.3f}")

# WHEN TO USE POLYNOMIAL REGRESSION:
# When you suspect non-linear relationships (diminishing returns, exponential growth)
# Degree 2 adds squared terms, degree 3 adds cubed terms, etc.
# Be careful: higher degrees can overfit easily!

# REGULARIZED REGRESSION - "Preventing overfitting"
# Problem: With lots of features, models can memorize training data but fail on new data
# Solution: Add penalties for large coefficients to keep the model simpler
print("\nRegularized Regression:")

# Ridge Regression (L2 regularization) - shrinks coefficients toward zero
ridge = Ridge(alpha=1.0)  # alpha controls penalty strength
ridge.fit(X_multiple, y)
print(f"Ridge R-squared: {ridge.score(X_multiple, y):.3f}")

# Lasso Regression (L1 regularization) - can set coefficients exactly to zero
lasso = Lasso(alpha=1.0)
lasso.fit(X_multiple, y)
print(f"Lasso R-squared: {lasso.score(X_multiple, y):.3f}")
print("Lasso feature selection (non-zero coefficients):")
for name, coef in zip(feature_names, lasso.coef_):
    if abs(coef) > 0.01:
        print(f"  {name}: {coef:,.2f}")

# RIDGE VS LASSO:
# Ridge: Keeps all features but makes coefficients smaller (good for correlated features)
# Lasso: Can eliminate features entirely (automatic feature selection)
# Higher alpha = more regularization = simpler model
# Use cross-validation to find optimal alpha value


# -----------------------------------------------------------------------------
# 🎯 CLASSIFICATION MODELS - Predicting Categories
# -----------------------------------------------------------------------------

# WHAT IS CLASSIFICATION?
# Instead of predicting numbers (regression), we predict categories or classes.
# Examples: spam/not spam, approve/reject loan, high/low performer
#
# OUR SCENARIO: Can we predict if an employee will be a high performer based on
# their age, experience, education, and salary? This is a binary classification
# problem (yes/no, 1/0).

print("\n" + "-" * 60)
print("CLASSIFICATION MODELS")
print("-" * 60)

# PREPARE CLASSIFICATION DATA
# We need to split our data BEFORE training to get honest performance estimates
# Never evaluate your model on the same data you trained it on!
X_class = employees_df[["age", "years_experience", "education_level", "salary"]]
y_class = employees_df["high_performer"]  # 1 = high performer, 0 = not

# Split data for proper evaluation (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

# WHY TRAIN-TEST SPLIT?
# Training data: Model learns patterns from this
# Test data: We evaluate performance on this (model has never seen it)
# This simulates how the model will perform on future, unknown data

# LOGISTIC REGRESSION - "Linear regression's classification cousin"
# Despite the name, this is for classification! It predicts probabilities between 0 and 1,
# then converts to classes. Uses the logistic function to squash outputs to 0-1 range.
print("Logistic Regression (Predicting High Performers):")
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)  # Predicted classes (0 or 1)
y_pred_proba = log_reg.predict_proba(X_test)[
    :, 1
]  # Predicted probabilities for class 1

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.3f}")

# HOW LOGISTIC REGRESSION WORKS:
# 1. Takes linear combination of features (like linear regression)
# 2. Applies logistic function to convert to probability (0 to 1)
# 3. If probability > 0.5, predicts class 1; otherwise class 0
# Coefficients tell you how each feature affects the log-odds of being class 1

# DECISION TREE CLASSIFIER - "A series of yes/no questions"
# Decision trees work like a flowchart: "Is salary > $70k? If yes, is experience > 5 years?"
# They're easy to interpret but can overfit easily.
print("\nDecision Tree Classifier:")
tree_clf = DecisionTreeClassifier(
    random_state=42, max_depth=4
)  # Limit depth to prevent overfitting
tree_clf.fit(X_train, y_train)

y_pred_tree = tree_clf.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_tree):.3f}")

# Feature importance from decision tree - which features matter most for splitting?
print("Feature Importance (Decision Tree):")
for name, importance in zip(X_class.columns, tree_clf.feature_importances_):
    print(f"  {name}: {importance:.3f}")

# DECISION TREE ADVANTAGES:
# - Easy to interpret (you can draw the decision flow)
# - Handles non-linear relationships naturally
# - No need to scale features
# DISADVANTAGES:
# - Can overfit easily (use max_depth, min_samples_split to control)
# - Small changes in data can create very different trees

# CONFUSION MATRIX AND CLASSIFICATION METRICS
# A confusion matrix shows you exactly where your model makes mistakes
print("\nConfusion Matrix (Logistic Regression):")
cm = confusion_matrix(y_test, y_pred_log)
print(cm)
print("Format: [True Negatives, False Positives]")
print("        [False Negatives, True Positives]")

# UNDERSTANDING THE CONFUSION MATRIX:
# True Negatives (TN): Correctly predicted "not high performer"
# False Positives (FP): Incorrectly predicted "high performer" (Type I error)
# False Negatives (FN): Incorrectly predicted "not high performer" (Type II error)
# True Positives (TP): Correctly predicted "high performer"

print("\nClassification Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.3f}")  # (TP + TN) / Total
print(f"Precision: {precision_score(y_test, y_pred_log):.3f}")  # TP / (TP + FP)
print(f"Recall (Sensitivity): {recall_score(y_test, y_pred_log):.3f}")  # TP / (TP + FN)
print(
    f"F1-Score: {f1_score(y_test, y_pred_log):.3f}"
)  # 2 * (Precision * Recall) / (Precision + Recall)

# Specificity calculation - True Negative Rate
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)  # TN / (TN + FP)
print(f"Specificity: {specificity:.3f}")

# METRIC INTERPRETATIONS:
# Accuracy: Overall correctness (can be misleading with imbalanced classes)
# Precision: Of those predicted as high performers, how many actually are?
# Recall: Of actual high performers, how many did we catch?
# F1-Score: Harmonic mean of precision and recall (good single metric)
# Specificity: Of actual non-high performers, how many did we correctly identify?

# ROC CURVE AND AUC - "How well can we separate the classes?"
# ROC curve shows the tradeoff between True Positive Rate and False Positive Rate
# at different probability thresholds
print("\nROC Analysis:")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"AUC (Area Under Curve): {roc_auc:.3f}")

# Plot ROC Curve for visual understanding
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("ROC Curve: Higher AUC = Better classifier (1.0 = perfect, 0.5 = random)")

# UNDERSTANDING ROC AND AUC:
# ROC curve: Shows how the classifier performs at all possible thresholds
# AUC = 1.0: Perfect classifier (can separate classes completely)
# AUC = 0.5: Random guessing (no discriminative ability)
# AUC < 0.5: Worse than random (but you could flip predictions!)
# Good rule of thumb: AUC > 0.8 is pretty good, AUC > 0.9 is excellent


# -----------------------------------------------------------------------------
# 🎭 CLUSTERING - Finding Hidden Groups in Data
# -----------------------------------------------------------------------------

# WHAT IS CLUSTERING?
# It's unsupervised learning - finding patterns without being told what to look for.
# We give the algorithm data and say "find groups that are similar to each other."
# Real world uses: Customer segmentation, gene analysis, market research
#
# OUR SCENARIO: Can we find natural groups of employees based on their characteristics?
# Maybe we'll discover "young high earners," "experienced steady performers," etc.

print("\n" + "-" * 60)
print("CLUSTERING")
print("-" * 60)

# PREPARE DATA FOR CLUSTERING
# Use only numerical features - clustering algorithms work with distances
X_cluster = employees_df[["age", "years_experience", "salary", "performance_score"]]

# CRITICAL: Standardize features so they're on the same scale
# Without this, salary (in thousands) would dominate age (in tens)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# WHY STANDARDIZATION FOR CLUSTERING?
# K-means uses Euclidean distance. If one feature has much larger values,
# it will dominate the distance calculation. Standardizing puts all features
# on equal footing (mean=0, std=1).

# K-MEANS CLUSTERING - "Group similar things together"
# K-means finds k clusters by minimizing within-cluster distances
# We need to specify k in advance (that's a limitation)
print("K-Means Clustering (k=3):")
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels back to our dataframe to interpret results
employees_df["cluster"] = clusters

print("Cluster Centers (original scale):")
centers_original = scaler.inverse_transform(
    kmeans.cluster_centers_
)  # Convert back to original units
feature_names = ["age", "years_experience", "salary", "performance_score"]

for i, center in enumerate(centers_original):
    print(f"Cluster {i}:")
    for name, value in zip(feature_names, center):
        print(f"  {name}: {value:.1f}")

print("\nCluster Sizes:")
for i in range(3):
    size = sum(clusters == i)
    print(f"Cluster {i}: {size} employees ({size / len(employees_df) * 100:.1f}%)")

# HOW K-MEANS WORKS:
# 1. Randomly place k cluster centers
# 2. Assign each point to nearest center
# 3. Move centers to average of assigned points
# 4. Repeat until centers stop moving
#
# INTERPRETING CLUSTERS:
# Look at the cluster centers to understand what each group represents.
# Cluster 0 might be "young, inexperienced, lower salary"
# Cluster 1 might be "experienced, high performers, high salary"


# -----------------------------------------------------------------------------
# ⚖️ BIAS-VARIANCE TRADEOFF - Understanding Model Performance
# -----------------------------------------------------------------------------

# THE FUNDAMENTAL ML TRADEOFF:
# Every model faces a choice between being too simple (high bias) or too complex (high variance)
#
# BIAS: How far off is your model from the true relationship?
# - High bias = underfitting = model is too simple to capture patterns
# - Like trying to fit a curve with a straight line
#
# VARIANCE: How much does your model change with different training data?
# - High variance = overfitting = model memorizes training data noise
# - Like fitting a super wiggly line that goes through every training point
#
# GOAL: Find the sweet spot that minimizes total error = bias² + variance + noise

print("\n" + "-" * 60)
print("BIAS-VARIANCE TRADEOFF")
print("-" * 60)

# DEMONSTRATING BIAS-VARIANCE WITH POLYNOMIAL REGRESSION
# We'll fit polynomials of different complexities to show the tradeoff
print("Bias-Variance Demo with Polynomial Regression:")

# Create simple dataset for clear demonstration
np.random.seed(42)
X_demo = np.linspace(0, 1, 50).reshape(-1, 1)
y_true = 1.5 * X_demo.ravel() + 0.5 * np.sin(
    2 * np.pi * X_demo.ravel()
)  # True relationship
y_demo = y_true + np.random.normal(0, 0.1, X_demo.shape[0])  # Add noise

degrees = [1, 4, 15]
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly_demo = poly.fit_transform(X_demo)

    model = LinearRegression()
    model.fit(X_poly_demo, y_demo)

    score = model.score(X_poly_demo, y_demo)

    if degree == 1:
        print(f"Degree {degree} (High Bias, Low Variance): R² = {score:.3f}")
        print("  → Too simple, misses the curve, but consistent across datasets")
    elif degree == 4:
        print(f"Degree {degree} (Balanced): R² = {score:.3f}")
        print("  → Just right complexity, captures main pattern without overfitting")
    else:
        print(f"Degree {degree} (Low Bias, High Variance): R² = {score:.3f}")
        print("  → Too complex, fits training noise, would vary wildly on new data")

# KEY INSIGHT:
# Notice how R² keeps increasing, but degree 15 would probably perform terribly
# on new data despite perfect training performance. This is overfitting!


# -----------------------------------------------------------------------------
# 🔍 MODEL EVALUATION - Proper Testing and Validation
# -----------------------------------------------------------------------------

# THE CARDINAL RULE: Never evaluate a model on data it was trained on!
#
# WHY? Because models can memorize training data. A model that perfectly fits
# training data might be useless on new data. We need honest performance estimates.
#
# THE SOLUTION: Always hold out some data for testing, and use cross-validation
# to get robust estimates of model performance.

print("\n" + "-" * 60)
print("MODEL EVALUATION")
print("-" * 60)

# TRAIN-TEST SPLIT - "The most basic honest evaluation"
print("Train-Test Split:")
X_eval = employees_df[["years_experience", "education_level"]]
y_eval = employees_df["salary"]

X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
    X_eval, y_eval, test_size=0.2, random_state=42
)

model_eval = LinearRegression()
model_eval.fit(X_train_eval, y_train_eval)  # Train on 80% of data

train_score = model_eval.score(
    X_train_eval, y_train_eval
)  # Performance on training data
test_score = model_eval.score(X_test_eval, y_test_eval)  # Performance on unseen data

print(f"Training R²: {train_score:.3f}")
print(f"Testing R²: {test_score:.3f}")

if train_score - test_score > 0.1:
    print("Warning: Large gap suggests overfitting")
    print("Model memorized training data but doesn't generalize well")
else:
    print("Good: Model generalizes well")
    print("Small gap means performance is consistent on new data")

# INTERPRETING THE GAP:
# Small gap: Model generalizes well
# Large gap (train >> test): Overfitting - model memorized training noise
# Test > train: Usually indicates a lucky test set or data leakage

# CROSS-VALIDATION - "More robust than a single train-test split"
# Problem with single split: What if you got lucky/unlucky with your test set?
# Solution: Split data multiple ways and average the results
print("\nCross-Validation (5-fold):")
cv_scores = cross_val_score(model_eval, X_eval, y_eval, cv=5, scoring="r2")
print(f"CV Scores: {[f'{score:.3f}' for score in cv_scores]}")
print(f"Mean CV Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

# HOW 5-FOLD CV WORKS:
# 1. Split data into 5 equal parts
# 2. Train on 4 parts, test on 1 part (repeat 5 times)
# 3. Average the 5 performance scores
#
# BENEFITS:
# - More reliable estimate (uses all data for both training and testing)
# - Shows performance variability (standard deviation)

# FEATURE SCALING COMPARISON - "Does scaling matter for this algorithm?"
print("\nFeature Scaling Comparison:")

# Original scale
log_reg_original = LogisticRegression(random_state=42)
original_score = cross_val_score(log_reg_original, X_train, y_train, cv=3).mean()

# StandardScaler
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
log_reg_std = LogisticRegression(random_state=42)
std_score = cross_val_score(log_reg_std, X_train_std, y_train, cv=3).mean()

# MinMaxScaler
scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
log_reg_minmax = LogisticRegression(random_state=42)
minmax_score = cross_val_score(log_reg_minmax, X_train_minmax, y_train, cv=3).mean()

print(f"No scaling: {original_score:.3f}")
print(f"StandardScaler: {std_score:.3f}")
print(f"MinMaxScaler: {minmax_score:.3f}")

# WHY SCALING MATTERS:
# Algorithms like logistic regression, SVM, k-means are sensitive to feature scales
# Tree-based algorithms (random forest, decision trees) don't need scaling
# Always scale for: logistic regression, SVM, neural networks, k-means
# Scaling optional for: decision trees, random forest, gradient boosting


# -----------------------------------------------------------------------------
# 📊 MODEL COMPARISON - Choosing the Best Approach
# -----------------------------------------------------------------------------

print("\n" + "-" * 60)
print("MODEL COMPARISON")
print("-" * 60)

# COMPARE MULTIPLE REGRESSION MODELS
# It's good practice to try several algorithms and pick the best performer
print("Regression Model Comparison:")
models_reg = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
}

for name, model in models_reg.items():
    scores = cross_val_score(model, X_multiple, y, cv=5, scoring="r2")
    print(f"{name:10s}: {scores.mean():.3f} (±{scores.std():.3f})")

# COMPARE CLASSIFICATION MODELS
print("\nClassification Model Comparison:")
models_clf = {
    "Logistic": LogisticRegression(random_state=42),
    "Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
}

X_scaled_clf = StandardScaler().fit_transform(X_train)
for name, model in models_clf.items():
    scores = cross_val_score(model, X_scaled_clf, y_train, cv=3, scoring="accuracy")
    print(f"{name:10s}: {scores.mean():.3f} (±{scores.std():.3f})")

# MODEL SELECTION TIPS:
# - Cross-validation gives more reliable estimates than single train-test split
# - Look at both mean performance AND standard deviation (consistency)
# - Consider interpretability vs performance tradeoffs
# - Try multiple algorithms - no single best algorithm for all problems


# -----------------------------------------------------------------------------
# 🎯 PRACTICAL DATA SCIENCE WORKFLOW - Putting It All Together
# -----------------------------------------------------------------------------

print("\n" + "-" * 60)
print("PRACTICAL DATA SCIENCE WORKFLOW")
print("-" * 60)

# REAL-WORLD DATA SCIENCE ISN'T JUST ABOUT ALGORITHMS
# It's about following a systematic process that ensures reliable, actionable results.
# This workflow shows how all the pieces fit together in a complete project.

# Demonstrate a complete ML workflow from start to finish
print("Complete ML Pipeline Example:")
print("Problem: Predict employee high performance")

# 1. DATA EXPLORATION - "Know your data before you model it"
print("\n1. DATA EXPLORATION:")
print(f"   Dataset shape: {employees_df.shape}")
print(f"   Missing values: {employees_df.isnull().sum().sum()}")
print(
    f"   Target distribution: {employees_df['high_performer'].value_counts().to_dict()}"
)

# WHY EXPLORE FIRST?
# - Understand what you're working with
# - Spot data quality issues early
# - Check for class imbalance that might affect modeling

# 2. FEATURE ENGINEERING - "Create better inputs for your model"
print("\n2. FEATURE ENGINEERING:")
# Create interaction feature - sometimes combinations matter more than individual features
employees_df["experience_education"] = (
    employees_df["years_experience"] * employees_df["education_level"]
)
# Create binned age groups for interpretability
employees_df["age_group"] = pd.cut(
    employees_df["age"],
    bins=[0, 30, 40, 50, 100],
    labels=["Young", "Mid", "Senior", "Veteran"],
)
print("   ✅ Created interaction feature: experience × education")
print("   ✅ Created age groups for interpretability")

# FEATURE ENGINEERING IS OFTEN THE MOST IMPACTFUL STEP
# Good features can make a simple model outperform a complex model with poor features

# 3. MODEL PIPELINE - "Systematic approach to model building"
print("\n3. MODEL PIPELINE:")
X_pipeline = employees_df[
    ["age", "years_experience", "education_level", "salary", "experience_education"]
]
y_pipeline = employees_df["high_performer"]

# Split data with stratification to maintain class balance
X_train_pipe, X_test_pipe, y_train_pipe, y_test_pipe = train_test_split(
    X_pipeline, y_pipeline, test_size=0.2, stratify=y_pipeline, random_state=42
)

# Scale features (important for logistic regression)
scaler_pipe = StandardScaler()
X_train_scaled = scaler_pipe.fit_transform(X_train_pipe)
X_test_scaled = scaler_pipe.transform(X_test_pipe)  # Use same scaler, don't refit!

# Train final model
final_model = LogisticRegression(random_state=42)
final_model.fit(X_train_scaled, y_train_pipe)

# 4. MODEL EVALUATION - "How good is good enough?"
print("\n4. MODEL EVALUATION:")
y_pred_final = final_model.predict(X_test_scaled)
y_pred_proba_final = final_model.predict_proba(X_test_scaled)[:, 1]

print(f"   Final Accuracy: {accuracy_score(y_test_pipe, y_pred_final):.3f}")
print(f"   Final F1-Score: {f1_score(y_test_pipe, y_pred_final):.3f}")
print(f"   Final AUC: {auc(*roc_curve(y_test_pipe, y_pred_proba_final)[:2]):.3f}")

# EVALUATION SHOULD MATCH YOUR BUSINESS OBJECTIVE
# High precision if false positives are costly
# High recall if false negatives are costly
# Balanced F1 if both matter equally

# 5. BUSINESS INSIGHTS - "What does this mean for decision-makers?"
print("\n5. BUSINESS INSIGHTS:")
print("   Top factors for high performance:")
feature_importance = abs(final_model.coef_[0])
feature_names_pipe = [
    "age",
    "years_experience",
    "education_level",
    "salary",
    "experience_education",
]
for name, importance in sorted(
    zip(feature_names_pipe, feature_importance), key=lambda x: x[1], reverse=True
)[:3]:
    print(f"   • {name}: {importance:.3f}")

# 6. DEPLOYMENT READINESS CHECKLIST
print("\n6. DEPLOYMENT READINESS:")
print("   ✅ Model trained on clean, representative data")
print("   ✅ Cross-validated performance metrics documented")
print("   ✅ Feature scaling pipeline preserved for production")
print("   ✅ Business insights extracted for stakeholders")

# WHAT'S MISSING FOR PRODUCTION?
# - Model monitoring and drift detection
# - A/B testing framework
# - Fallback strategies if model fails
# - Documentation and handover procedures


# -----------------------------------------------------------------------------
# 🚦 BEST PRACTICES & COMMON PITFALLS
# -----------------------------------------------------------------------------

# 1️⃣ Data Leakage 🚱:
#    - Never let info from the future or the target sneak into your features!
#    - Always split your data into train/test BEFORE any preprocessing. Leakage = cheating.

# 2️⃣ Overfitting 🎭:
#    - If your model is too complex, it might just memorize the data (and fail on new stuff).
#    - Use cross-validation, keep models simple, and try regularization to help your model generalize.

# 3️⃣ Feature Scaling 📏:
#    - Some algorithms (like k-means, logistic regression) care about feature scale.
#    - Fit your scaler ONLY on training data, then transform both train & test. No peeking!

# 4️⃣ Correlation ≠ Causation 🔗❌:
#    - Just because two things move together doesn’t mean one causes the other.
#    - Use domain knowledge and, if you can, experiments to figure out what’s really going on.

# 5️⃣ Statistical vs Practical Significance 📉 vs 💡:
#    - A tiny p-value (< 0.05) might be “statistically significant” but is it actually useful?
#    - Always check the effect size and ask: does this matter in the real world?

# 6️⃣ Sample Size Matters 🧮:
#    - Small samples = big risk of weird, unreliable results.
#    - Try to get enough data so your findings actually mean something!


# -----------------------------------------------------------------------------
# 📝 COURSE RECAP - Everything We Covered
# -----------------------------------------------------------------------------

# Topics Covered in this Cheatsheet:
#   ✅ Descriptive Statistics (mean, median, mode, std, variance)
#   ✅ Inferential Statistics (confidence intervals, CLT, standard error)
#   ✅ Hypothesis Testing (t-tests, chi-square, ANOVA)
#   ✅ Regression (linear, multiple, polynomial, regularized)
#   ✅ Classification (logistic regression, decision trees)
#   ✅ Model Evaluation (confusion matrix, precision, recall, F1, ROC/AUC)
#   ✅ Clustering (k-means)
#   ✅ ML Concepts (bias-variance, overfitting, cross-validation)
#   ✅ Feature Scaling (StandardScaler, MinMaxScaler)
#   ✅ Model Comparison and Selection
#   ✅ Complete ML Pipeline (end-to-end workflow)
#   ✅ Best Practices and Common Pitfalls


print("\nCongratulations! You've completed a comprehensive tour of")
print("statistics and machine learning fundamentals.")

print("\n" + "-" * 60)
print("END OF STATISTICS & ML CHEATSHEET")
print("-" * 60)
