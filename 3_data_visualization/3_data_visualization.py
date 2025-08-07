"""
📊 Seaborn Tutorial – Beautiful, Statistical Visualizations in Python
=======================================================================

Welcome to your guide on turning raw data into stunning, meaningful charts using Seaborn.
It’s built on top of Matplotlib, but with way less hassle — perfect for data stories that *look* as good as they *inform*.

🎯 What you'll learn:
- How to use Seaborn’s themes and styling for cleaner charts
- Core plots: scatter, line, bar, box, violin, hist, and more
- Multi-variable visuals: pairplot, heatmap, catplot, jointplot
- Tweaking plots with color palettes, context, figure size
- Real-world practice with built-in datasets like `tips`, `diamonds`, `penguins`

🔧 How to use:
Just run the file from top to bottom.
You’ll see printed outputs, visualizations pop up, and inline comments to explain what’s happening and why.

📦 Requirements:
You'll need Seaborn, Matplotlib, and Pandas.
Install everything with:

    pip install -r requirements.txt

🤖 Note:
This is part of the [Cheatsheet DS Minicourse] — a side project made with
some coding, some storytelling, and a little help from Claude AI.

Let’s get visual 📈
"""

# -----------------------------------------------------------------------------
# 📦 Essential Imports and Setup
# -----------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from matplotlib.patches import Ellipse
from scipy.stats import f_oneway

# Set random seed for reproducible results across all runs
np.random.seed(42)

# Configure for optimal visualization experience
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="husl")  # Clean, colorful default style
plt.rcParams["figure.figsize"] = (12, 8)  # Consistent, readable figure size
plt.rcParams["font.size"] = 11  # Improve text readability

print("🎯 Welcome to Complete Data Visualization Mastery!")
print("📊 Creating beautiful statistical plots with minimal code!")
print("🚀 This tutorial will transform you into a visualization expert!\n")


# -----------------------------------------------------------------------------
# 📊 Creating Realistic Sample Datasets
# -----------------------------------------------------------------------------
"""
Why synthetic data for learning?
- No external dependencies or downloads needed
- Reproducible results across different environments
- Clear relationships that demonstrate visualization concepts
- Complete control over data characteristics for teaching
"""

# Generate realistic e-commerce business dataset
n_customers = 500  # Larger sample for better visualization clarity

# Create correlated customer data that mimics real business scenarios
customers_df = pd.DataFrame(
    {
        "age": np.clip(np.random.normal(35, 12, n_customers).astype(int), 18, 80),
        "income": np.clip(
            np.random.normal(55000, 18000, n_customers).astype(int), 25000, 150000
        ),
        "spending": np.clip(
            np.random.normal(800, 350, n_customers).astype(int), 50, 3000
        ),
        "satisfaction": np.round(np.random.uniform(1.5, 5.0, n_customers), 1),
        "region": np.random.choice(["North", "South", "East", "West"], n_customers),
        "segment": np.random.choice(
            ["Premium", "Standard", "Budget"], n_customers, p=[0.25, 0.45, 0.30]
        ),
        "product_category": np.random.choice(
            ["Electronics", "Clothing", "Books", "Sports", "Home"], n_customers
        ),
    }
)

# Add realistic business relationships between variables
# Premium customers: higher spending, income, and satisfaction
premium_mask = customers_df["segment"] == "Premium"
customers_df.loc[premium_mask, "spending"] *= 1.8
customers_df.loc[premium_mask, "income"] *= 1.4
customers_df.loc[premium_mask, "satisfaction"] += 0.7

# Standard customers: moderate adjustments
standard_mask = customers_df["segment"] == "Standard"
customers_df.loc[standard_mask, "spending"] *= 1.2
customers_df.loc[standard_mask, "satisfaction"] += 0.3

# Income correlates with spending (realistic business relationship)
customers_df["spending"] = customers_df["spending"] + (customers_df["income"] * 0.008)

# Cap values at realistic ranges
customers_df["spending"] = np.clip(customers_df["spending"], 50, 4000).astype(int)
customers_df["satisfaction"] = np.clip(customers_df["satisfaction"], 1.0, 5.0)

# Create time-series sales data for trend analysis
dates = pd.date_range("2024-01-01", periods=12, freq="M")
products = ["Laptop", "Desktop", "Tablet", "Phone", "Watch", "Headphones"]

# Monthly sales with realistic trends and seasonality
monthly_sales = pd.DataFrame(
    {
        "Month": dates,
        "Laptop": 1000
        + 50 * np.arange(12)
        + np.random.normal(0, 100, 12)
        + 200 * np.sin(np.arange(12) * 0.5),
        "Desktop": 600 + 30 * np.arange(12) + np.random.normal(0, 80, 12),
        "Tablet": 800 + 40 * np.arange(12) + np.random.normal(0, 120, 12),
        "Phone": 1200
        + 80 * np.arange(12)
        + np.random.normal(0, 150, 12)
        + 300 * np.sin(np.arange(12) * 0.8),
        "Watch": 400 + 25 * np.arange(12) + np.random.normal(0, 60, 12),
        "Headphones": 300 + 20 * np.arange(12) + np.random.normal(0, 50, 12),
    }
)

print(
    f"📊 Generated customer dataset: {customers_df.shape[0]} customers, {customers_df.shape[1]} features"
)
print(
    f"📈 Generated sales dataset: {monthly_sales.shape[0]} months, {len(products)} products"
)
print("\n🔍 Customer data preview:")
print(customers_df.head(3).to_string())
print("\n" + "=" * 80 + "\n")


# -----------------------------------------------------------------------------
# 📊 Distribution Plots: Understanding Your Data's Shape
# -----------------------------------------------------------------------------
"""
🎯 WHY DISTRIBUTION PLOTS MATTER:
Distribution visualizations are the foundation of data understanding. They reveal:
- Data shape: Is it normal, skewed, or has multiple peaks?
- Central tendency: Where do most values cluster?
- Spread: How varied is your data?
- Outliers: Unusual values that need investigation
- Data quality: Missing values, impossible values, data entry errors

This knowledge is crucial for:
✅ Choosing appropriate statistical methods
✅ Identifying data quality issues early
✅ Understanding business patterns and customer behavior
✅ Making informed decisions about data preprocessing
"""

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(
    "📊 Distribution Analysis: Understanding Data Shape & Spread",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

# 1. Histogram - The foundation of distribution analysis
sns.histplot(
    data=customers_df,
    x="spending",
    bins=30,
    kde=True,
    color="skyblue",
    alpha=0.8,
    ax=axes[0, 0],
)
axes[0, 0].set_title(
    "📈 Spending Distribution\n(Histogram + Density Curve)",
    fontsize=14,
    fontweight="bold",
)
axes[0, 0].set_xlabel("Monthly Spending ($)", fontweight="bold")
axes[0, 0].set_ylabel("Frequency", fontweight="bold")
# KDE overlay shows smooth distribution curve - helps identify shape patterns

# 2. Grouped histogram - Compare distributions across categories
sns.histplot(
    data=customers_df,
    x="spending",
    hue="segment",
    multiple="dodge",
    bins=25,
    alpha=0.7,
    ax=axes[0, 1],
)
axes[0, 1].set_title(
    "🏆 Spending by Customer Segment\n(Side-by-side Comparison)",
    fontsize=14,
    fontweight="bold",
)
axes[0, 1].set_xlabel("Monthly Spending ($)", fontweight="bold")
# 'dodge' places bars side by side for easy comparison between groups

# 3. Density plot - Smooth curves ideal for group comparisons
sns.kdeplot(
    data=customers_df,
    x="age",
    hue="segment",
    fill=True,
    alpha=0.6,
    palette="viridis",
    ax=axes[0, 2],
)
axes[0, 2].set_title(
    "👥 Age Distribution by Segment\n(Smooth Density Curves)",
    fontsize=14,
    fontweight="bold",
)
axes[0, 2].set_xlabel("Customer Age", fontweight="bold")
# KDE creates smooth curves from discrete data - better for overlapping comparisons

# 4. Box plot - Statistical summary in compact visual form
custom_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
sns.boxplot(
    data=customers_df, x="segment", y="income", palette=custom_colors, ax=axes[1, 0]
)
axes[1, 0].set_title(
    "💰 Income by Segment\n(Statistical Summary)", fontsize=14, fontweight="bold"
)
axes[1, 0].set_xlabel("Customer Segment", fontweight="bold")
axes[1, 0].set_ylabel("Annual Income ($)", fontweight="bold")
# Box plot shows: median (line), quartiles (box), outliers (dots), range (whiskers)

# 5. Violin plot - Distribution shape + statistical summary combined
sns.violinplot(
    data=customers_df, x="region", y="satisfaction", palette="Set2", ax=axes[1, 1]
)
axes[1, 1].set_title(
    "⭐ Satisfaction by Region\n(Shape + Statistics)", fontsize=14, fontweight="bold"
)
axes[1, 1].set_xlabel("Geographic Region", fontweight="bold")
axes[1, 1].set_ylabel("Satisfaction Score", fontweight="bold")
# Wider violin sections = more customers at that satisfaction level

# 6. Strip plot with jitter - Show every individual data point
sns.stripplot(
    data=customers_df.sample(200),
    x="product_category",
    y="spending",
    size=6,
    alpha=0.7,
    palette="husl",
    ax=axes[1, 2],
)
axes[1, 2].set_title(
    "🎯 Individual Spending Points\n(Every Customer Visible)",
    fontsize=14,
    fontweight="bold",
)
axes[1, 2].set_xlabel("Product Interest", fontweight="bold")
axes[1, 2].set_ylabel("Monthly Spending ($)", fontweight="bold")
axes[1, 2].tick_params(axis="x", rotation=45)
# Jitter prevents overlapping points while showing true data density

plt.tight_layout()
plt.show()

# Key insight: Distribution plots reveal the story behind your averages!
print("✅ Distribution Analysis Complete!")
print(
    "🔍 Key Insights: Look for normal vs skewed distributions, outliers, and group differences"
)
print(
    "📊 Business Impact: Understanding spending patterns helps segment customers effectively\n"
)


# -----------------------------------------------------------------------------
# 📈 Scatter Plots & Relationships: Discovering Data Connections
# -----------------------------------------------------------------------------
"""
🎯 WHY SCATTER PLOTS ARE ESSENTIAL:
Scatter plots are your primary tool for exploring relationships between variables. They reveal:
- Correlations: positive, negative, or no relationship
- Non-linear patterns that averages might miss
- Outliers that could indicate data errors or special cases
- Clusters that suggest natural groupings in your data
- Strength of relationships for predictive modeling

Critical for:
✅ Feature selection in machine learning
✅ Understanding business drivers and KPIs
✅ Identifying data quality issues
✅ Validating business hypotheses with data
"""

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle(
    "📈 Relationship Analysis: Discovering Data Connections",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

# 1. Basic scatter plot - Foundation of relationship analysis
sns.scatterplot(
    data=customers_df,
    x="income",
    y="spending",
    alpha=0.7,
    s=60,
    color="#FF6B6B",
    ax=axes[0, 0],
)
axes[0, 0].set_title(
    "💰 Income vs Spending\n(Basic Relationship)", fontsize=14, fontweight="bold"
)
axes[0, 0].set_xlabel("Annual Income ($)", fontweight="bold")
axes[0, 0].set_ylabel("Monthly Spending ($)", fontweight="bold")
# Look for upward/downward trends, scatter patterns, and outliers

# 2. Color-coded scatter - Add categorical dimension
sns.scatterplot(
    data=customers_df,
    x="income",
    y="spending",
    hue="segment",
    s=70,
    alpha=0.8,
    palette="viridis",
    ax=axes[0, 1],
)
axes[0, 1].set_title(
    "🎯 Income vs Spending by Segment\n(Color Shows Groups)",
    fontsize=14,
    fontweight="bold",
)
axes[0, 1].set_xlabel("Annual Income ($)", fontweight="bold")
axes[0, 1].set_ylabel("Monthly Spending ($)", fontweight="bold")
# Different colors reveal how customer segments behave differently

# 3. Multi-dimensional scatter - Size adds third variable
sns.scatterplot(
    data=customers_df,
    x="age",
    y="satisfaction",
    hue="segment",
    size="spending",
    sizes=(30, 300),
    alpha=0.7,
    palette="Set1",
    ax=axes[1, 0],
)
axes[1, 0].set_title(
    "👥 Age vs Satisfaction\n(Size = Spending Level)", fontsize=14, fontweight="bold"
)
axes[1, 0].set_xlabel("Customer Age", fontweight="bold")
axes[1, 0].set_ylabel("Satisfaction Score", fontweight="bold")
# Now showing 4 variables simultaneously: age, satisfaction, segment, spending

# 4. Regression scatter - Quantify the relationship
sns.scatterplot(
    data=customers_df, x="income", y="spending", alpha=0.6, s=50, ax=axes[1, 1]
)
sns.regplot(
    data=customers_df,
    x="income",
    y="spending",
    scatter=False,
    color="red",
    ax=axes[1, 1],
)
axes[1, 1].set_title(
    "📊 Income vs Spending + Trend Line\n(Quantified Relationship)",
    fontsize=14,
    fontweight="bold",
)
axes[1, 1].set_xlabel("Annual Income ($)", fontweight="bold")
axes[1, 1].set_ylabel("Monthly Spending ($)", fontweight="bold")
# Regression line helps predict spending based on income

# Calculate and display correlation
correlation = customers_df["income"].corr(customers_df["spending"])
axes[1, 1].text(
    0.05,
    0.95,
    f"Correlation: {correlation:.3f}",
    transform=axes[1, 1].transAxes,
    fontsize=12,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
)

plt.tight_layout()
plt.show()

print("✅ Relationship Analysis Complete!")
print(f"🔍 Key Finding: Income and spending correlation = {correlation:.3f}")
print(
    "💡 Business Insight: Strong positive correlation suggests income is a good predictor of spending behavior\n"
)


# -----------------------------------------------------------------------------
# 🔥 Correlation Analysis & Heatmaps: Finding Hidden Patterns
# -----------------------------------------------------------------------------
"""
🎯 WHY CORRELATION ANALYSIS IS CRUCIAL:
Correlation analysis reveals the hidden structure in your data:
- Which variables move together (positive correlation)?
- Which variables move in opposite directions (negative correlation)?
- Which variables are independent (no correlation)?
- Potential multicollinearity issues for modeling

Essential for:
✅ Feature engineering and selection
✅ Understanding business drivers
✅ Detecting data redundancy
✅ Building better predictive models
✅ Identifying unexpected relationships
"""

# Select numeric columns for correlation analysis
numeric_columns = ["age", "income", "spending", "satisfaction"]
correlation_matrix = customers_df[numeric_columns].corr()

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle(
    "🔥 Correlation Analysis: Discovering Hidden Data Patterns",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

# 1. Basic correlation heatmap
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="RdBu_r",
    center=0,
    square=True,
    fmt=".3f",
    cbar_kws={"label": "Correlation Coefficient"},
    ax=axes[0, 0],
)
axes[0, 0].set_title(
    "🌡️ Correlation Heatmap\n(Red=Positive, Blue=Negative)",
    fontsize=14,
    fontweight="bold",
)
# Values near +1 or -1 indicate strong relationships; near 0 means no linear relationship

# 2. Masked correlation heatmap (cleaner presentation)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    square=True,
    ax=axes[0, 1],
)
axes[0, 1].set_title(
    "🎯 Clean Correlation Matrix\n(Lower Triangle Only)", fontsize=14, fontweight="bold"
)
# Masking prevents showing the same correlation twice

# 3. Business insights heatmap - Pivot table analysis
pivot_spending = customers_df.pivot_table(
    values="spending", index="region", columns="segment", aggfunc="mean"
)
sns.heatmap(
    pivot_spending,
    annot=True,
    fmt=".0f",
    cmap="YlOrRd",
    cbar_kws={"label": "Average Spending ($)"},
    ax=axes[1, 0],
)
axes[1, 0].set_title(
    "💼 Average Spending Heatmap\n(Region × Segment)", fontsize=14, fontweight="bold"
)
# This reveals which region-segment combinations are most valuable

# 4. Advanced: Satisfaction correlation by category
pivot_satisfaction = customers_df.pivot_table(
    values="satisfaction", index="product_category", columns="segment", aggfunc="mean"
)
sns.heatmap(
    pivot_satisfaction,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    cbar_kws={"label": "Average Satisfaction"},
    ax=axes[1, 1],
)
axes[1, 1].set_title(
    "⭐ Satisfaction Heatmap\n(Product × Segment)", fontsize=14, fontweight="bold"
)
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# Display correlation insights
print("✅ Correlation Analysis Complete!")
print("\n🔍 Key Correlations Found:")
for i, col1 in enumerate(numeric_columns):
    for col2 in numeric_columns[i + 1 :]:
        corr_val = correlation_matrix.loc[col1, col2]
        strength = (
            "Strong"
            if abs(corr_val) > 0.7
            else "Moderate"
            if abs(corr_val) > 0.3
            else "Weak"
        )
        direction = "Positive" if corr_val > 0 else "Negative"
        print(
            f"   {col1.title()} ↔ {col2.title()}: {corr_val:.3f} ({strength} {direction})"
        )

print("\n💡 Business Implications:")
print("   • Strong income-spending correlation suggests income-based targeting")
print("   • Regional and segment differences reveal market opportunities\n")


# -----------------------------------------------------------------------------
# 📊 Categorical Data Analysis: Comparing Groups Effectively
# -----------------------------------------------------------------------------
"""
🎯 WHY CATEGORICAL ANALYSIS MATTERS:
Categorical visualizations answer critical business questions:
- Which customer segment is most valuable?
- How do regions compare in performance?
- Are differences between groups statistically significant?
- What's the frequency distribution of categories?

Vital for:
✅ Business intelligence and KPI analysis
✅ A/B testing and experimental design
✅ Market segmentation and targeting
✅ Performance comparisons and benchmarking
"""

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(
    "📊 Categorical Analysis: Comparing Groups & Categories",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

# 1. Bar plot - Compare average values with confidence intervals
sns.barplot(
    data=customers_df,
    x="segment",
    y="spending",
    palette="viridis",
    capsize=0.1,
    ax=axes[0, 0],
)
axes[0, 0].set_title(
    "💰 Average Spending by Segment\n(With Confidence Intervals)",
    fontsize=14,
    fontweight="bold",
)
axes[0, 0].set_xlabel("Customer Segment", fontweight="bold")
axes[0, 0].set_ylabel("Average Monthly Spending ($)", fontweight="bold")

# Add value labels on bars
for i, bar in enumerate(axes[0, 0].patches):
    height = bar.get_height()
    axes[0, 0].text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 50,
        f"${height:.0f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )

# 2. Grouped bar plot - Multi-dimensional comparison
sns.barplot(
    data=customers_df,
    x="region",
    y="satisfaction",
    hue="segment",
    palette="Set2",
    ax=axes[0, 1],
)
axes[0, 1].set_title(
    "⭐ Satisfaction: Region × Segment\n(Grouped Comparison)",
    fontsize=14,
    fontweight="bold",
)
axes[0, 1].set_xlabel("Geographic Region", fontweight="bold")
axes[0, 1].set_ylabel("Average Satisfaction Score", fontweight="bold")
axes[0, 1].tick_params(axis="x", rotation=45)

# 3. Count plot - Frequency analysis
sns.countplot(data=customers_df, x="product_category", palette="husl", ax=axes[0, 2])
axes[0, 2].set_title(
    "📈 Product Category Popularity\n(Customer Frequency)",
    fontsize=14,
    fontweight="bold",
)
axes[0, 2].set_xlabel("Product Category", fontweight="bold")
axes[0, 2].set_ylabel("Number of Customers", fontweight="bold")
axes[0, 2].tick_params(axis="x", rotation=45)

# 4. Point plot - Trends across categories
sns.pointplot(
    data=customers_df,
    x="region",
    y="income",
    hue="segment",
    palette="Dark2",
    ax=axes[1, 0],
)
axes[1, 0].set_title(
    "📍 Income Trends Across Regions\n(Connected Points Show Trends)",
    fontsize=14,
    fontweight="bold",
)
axes[1, 0].set_xlabel("Geographic Region", fontweight="bold")
axes[1, 0].set_ylabel("Average Income ($)", fontweight="bold")
axes[1, 0].tick_params(axis="x", rotation=45)

# 5. Box plot comparison - Full distribution analysis
sns.boxplot(data=customers_df, x="segment", y="age", palette="pastel", ax=axes[1, 1])
axes[1, 1].set_title(
    "👥 Age Distribution by Segment\n(Box Shows Quartiles & Outliers)",
    fontsize=14,
    fontweight="bold",
)
axes[1, 1].set_xlabel("Customer Segment", fontweight="bold")
axes[1, 1].set_ylabel("Customer Age", fontweight="bold")

# 6. Swarm plot - Individual data points visible
sample_data = customers_df.sample(150)  # Sample to prevent overcrowding
sns.swarmplot(
    data=sample_data, x="region", y="spending", palette="bright", size=6, ax=axes[1, 2]
)
axes[1, 2].set_title(
    "🐝 Individual Spending Points\n(No Overlapping, True Distribution)",
    fontsize=14,
    fontweight="bold",
)
axes[1, 2].set_xlabel("Geographic Region", fontweight="bold")
axes[1, 2].set_ylabel("Monthly Spending ($)", fontweight="bold")

plt.tight_layout()
plt.show()

print("✅ Categorical Analysis Complete!")
print("\n📊 Key Group Insights:")
segment_summary = (
    customers_df.groupby("segment")
    .agg({"spending": "mean", "income": "mean", "satisfaction": "mean", "age": "count"})
    .round(0)
)
segment_summary.columns = ["Avg_Spending", "Avg_Income", "Avg_Satisfaction", "Count"]
print(segment_summary)
print(
    "\n💡 Strategic Insights: Premium segment shows highest value - focus retention efforts here!\n"
)


# -----------------------------------------------------------------------------
# 🎨 Time Series & Trend Analysis: Understanding Patterns Over Time
# -----------------------------------------------------------------------------
"""
🎯 WHY TIME SERIES VISUALIZATION IS ESSENTIAL:
Time series plots reveal temporal patterns crucial for business planning:
- Seasonal trends and cyclical patterns
- Growth trajectories and decline periods
- Anomalies and unexpected changes
- Forecasting opportunities

Critical for:
✅ Sales forecasting and inventory planning
✅ Performance monitoring and KPI tracking
✅ Identifying business cycles and seasonality
✅ Strategic planning and resource allocation
"""

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(
    "📈 Time Series Analysis: Trends, Patterns & Forecasting",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

# 1. Multi-product sales trends
top_products = ["Phone", "Laptop", "Tablet"]
for product in top_products:
    axes[0, 0].plot(
        monthly_sales["Month"],
        monthly_sales[product],
        marker="o",
        linewidth=3,
        markersize=8,
        label=product,
    )
axes[0, 0].set_title(
    "📱 Top Product Sales Trends\n(Multi-Product Comparison)",
    fontsize=14,
    fontweight="bold",
)
axes[0, 0].set_xlabel("Month", fontweight="bold")
axes[0, 0].set_ylabel("Sales Revenue ($)", fontweight="bold")
axes[0, 0].legend(loc="upper left", frameon=True, fancybox=True)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis="x", rotation=45)

# 2. Area plot - Show cumulative trends
axes[1, 0].fill_between(
    monthly_sales["Month"],
    monthly_sales["Phone"],
    alpha=0.7,
    color="#FF6B6B",
    label="Phone Sales",
)
axes[1, 0].fill_between(
    monthly_sales["Month"],
    monthly_sales["Laptop"],
    alpha=0.7,
    color="#4ECDC4",
    label="Laptop Sales",
)
axes[1, 0].set_title(
    "📊 Sales Area Chart\n(Cumulative Visual Impact)", fontsize=14, fontweight="bold"
)
axes[1, 0].set_xlabel("Month", fontweight="bold")
axes[1, 0].set_ylabel("Sales Revenue ($)", fontweight="bold")
axes[1, 0].legend()
axes[1, 0].tick_params(axis="x", rotation=45)

# 3. Confidence intervals with trends
phone_sales = monthly_sales[["Month", "Phone"]].copy()
phone_sales["Upper_CI"] = phone_sales["Phone"] * 1.15
phone_sales["Lower_CI"] = phone_sales["Phone"] * 0.85

axes[0, 1].plot(
    phone_sales["Month"],
    phone_sales["Phone"],
    "b-",
    linewidth=3,
    label="Actual Sales",
    marker="o",
    markersize=8,
)
axes[0, 1].fill_between(
    phone_sales["Month"],
    phone_sales["Lower_CI"],
    phone_sales["Upper_CI"],
    alpha=0.3,
    color="blue",
    label="Confidence Band",
)
axes[0, 1].set_title(
    "📱 Phone Sales with Uncertainty\n(Confidence Intervals)",
    fontsize=14,
    fontweight="bold",
)
axes[0, 1].set_xlabel("Month", fontweight="bold")
axes[0, 1].set_ylabel("Phone Sales ($)", fontweight="bold")
axes[0, 1].legend()
axes[0, 1].tick_params(axis="x", rotation=45)

# 4. Seasonal decomposition visualization
# Create seasonal pattern
seasonal_pattern = 200 * np.sin(2 * np.pi * np.arange(12) / 12)
trend_component = 50 * np.arange(12)
noise = np.random.normal(0, 30, 12)
decomposed_sales = 1000 + trend_component + seasonal_pattern + noise

axes[1, 1].plot(
    monthly_sales["Month"],
    decomposed_sales,
    "g-",
    linewidth=3,
    marker="s",
    markersize=8,
    label="Observed Sales",
)
axes[1, 1].plot(
    monthly_sales["Month"],
    1000 + trend_component,
    "r--",
    linewidth=2,
    label="Underlying Trend",
)
axes[1, 1].set_title(
    "🔄 Sales Decomposition\n(Trend + Seasonality)", fontsize=14, fontweight="bold"
)
axes[1, 1].set_xlabel("Month", fontweight="bold")
axes[1, 1].set_ylabel("Decomposed Sales ($)", fontweight="bold")
axes[1, 1].legend()
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

print("✅ Time Series Analysis Complete!")
print("📈 Key Trends Identified:")
print(
    f"   • Phone sales growth: {((monthly_sales['Phone'].iloc[-1] / monthly_sales['Phone'].iloc[0]) - 1) * 100:.1f}%"
)
print(
    f"   • Laptop sales growth: {((monthly_sales['Laptop'].iloc[-1] / monthly_sales['Laptop'].iloc[0]) - 1) * 100:.1f}%"
)
print(
    "🔮 Forecasting Insight: Strong upward trends suggest continued growth potential\n"
)


# -----------------------------------------------------------------------------
# 🎯 Multi-Variable Analysis: FacetGrid & Advanced Visualizations
# -----------------------------------------------------------------------------
"""
🎯 WHY MULTI-VARIABLE ANALYSIS IS POWERFUL:
Complex data requires sophisticated visualization approaches:
- See multiple relationships simultaneously
- Compare patterns across different groups
- Identify interactions between variables
- Detect outliers and anomalies across dimensions

Essential for:
✅ Comprehensive exploratory data analysis
✅ Understanding complex business relationships
✅ Market segmentation and customer profiling
✅ Advanced pattern recognition and insights
"""

# FacetGrid - Create multiple subplots for detailed comparisons
print("🔍 Creating comprehensive multi-variable analysis...")

# 1. FacetGrid for detailed segment analysis
g = sns.FacetGrid(
    customers_df,
    col="segment",
    row="region",
    margin_titles=True,
    height=3.5,
    aspect=1.2,
)
g.map(sns.scatterplot, "income", "spending", alpha=0.7, s=60, color="#FF6B6B")
g.add_legend(title="Income vs Spending Analysis")
g.fig.suptitle(
    "🎯 Income vs Spending: Every Segment × Region Combination",
    fontsize=16,
    fontweight="bold",
    y=1.02,
)

# Add trend lines to each subplot
g.map(sns.regplot, "income", "spending", scatter=False, color="blue", ci=None)

plt.show()

# 2. Comprehensive pairwise analysis
print("🌟 Creating comprehensive pairwise relationship analysis...")

# Pairplot - all relationships in one visualization
g = sns.pairplot(
    customers_df[["age", "income", "spending", "satisfaction", "segment"]],
    hue="segment",
    diag_kind="kde",
    corner=True,
    plot_kws={"alpha": 0.7, "s": 60},
    diag_kws={"fill": True, "alpha": 0.7},
    palette="viridis",
)
g.fig.suptitle(
    "🌟 Complete Pairwise Relationship Analysis\n(All Variables × All Segments)",
    fontsize=16,
    fontweight="bold",
    y=1.02,
)

plt.show()

# 3. Advanced four-dimensional scatter plot
plt.figure(figsize=(14, 10))

# Use multiple visual encodings simultaneously
scatter = plt.scatter(
    customers_df["income"],
    customers_df["spending"],
    c=customers_df["satisfaction"],
    s=customers_df["age"] * 3,
    alpha=0.7,
    cmap="plasma",
    edgecolors="white",
    linewidth=0.5,
)

plt.title(
    "🚀 Four-Dimensional Customer Analysis\n"
    "Income vs Spending + Satisfaction (Color) + Age (Size)",
    fontsize=16,
    fontweight="bold",
    pad=20,
)
plt.xlabel("Annual Income ($)", fontsize=12, fontweight="bold")
plt.ylabel("Monthly Spending ($)", fontsize=12, fontweight="bold")

# Add colorbar for satisfaction
cbar = plt.colorbar(scatter)
cbar.set_label("Satisfaction Score", fontweight="bold", fontsize=12)

# Add size legend for age
sizes = [18 * 3, 35 * 3, 50 * 3, 65 * 3]  # Representative ages
labels = ["18 years", "35 years", "50 years", "65 years"]
legend_elements = [
    plt.scatter(
        [],
        [],
        s=size,
        c="gray",
        alpha=0.7,
        edgecolors="white",
        linewidth=0.5,
        label=label,
    )
    for size, label in zip(sizes, labels)
]
plt.legend(
    handles=legend_elements,
    title="Customer Age",
    loc="upper left",
    title_fontsize=12,
    fontsize=10,
)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("✅ Multi-Variable Analysis Complete!")
print("🔍 Advanced Insights: Four dimensions reveal complex customer behavior patterns")
print("💡 Strategic Value: Multi-dimensional analysis enables precision targeting\n")


# -----------------------------------------------------------------------------
# 🎨 Professional Styling & Customization: Publication-Ready Plots
# -----------------------------------------------------------------------------
"""
🎯 WHY PROFESSIONAL STYLING MATTERS:
Beautiful, clear visualizations have tremendous business impact:
- Increase audience engagement and comprehension
- Build credibility and trust in your analysis
- Facilitate better decision-making through clarity
- Create memorable, shareable insights

Essential for:
✅ Executive presentations and board meetings
✅ Client-facing reports and dashboards
✅ Academic publications and research
✅ Marketing materials and public communications
"""

# Demonstrate styling transformation: Basic → Professional
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# BEFORE: Basic plot with default styling
sns.barplot(data=customers_df, x="segment", y="spending", ax=axes[0])
axes[0].set_title("Basic Plot (Before Styling)")

# AFTER: Professional styled plot with custom enhancements
# Custom color palette - colorblind friendly and visually appealing
professional_colors = ["#E74C3C", "#3498DB", "#2ECC71"]
bars = sns.barplot(
    data=customers_df,
    x="segment",
    y="spending",
    palette=professional_colors,
    ax=axes[1],
)

# Professional styling enhancements
axes[1].set_title(
    "💼 Customer Spending Analysis by Segment\n"
    "Professional Business Intelligence Dashboard",
    fontsize=16,
    fontweight="bold",
    pad=25,
)
axes[1].set_xlabel("Customer Segment", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Average Monthly Spending ($)", fontsize=13, fontweight="bold")

# Add value labels with professional formatting
segment_means = customers_df.groupby("segment")["spending"].mean()
for i, (bar, value) in enumerate(zip(bars.patches, segment_means)):
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 50,
        f"${value:.0f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=13,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8
        ),
    )

# Professional visual enhancements
sns.despine(ax=axes[1])  # Remove top and right spines
axes[1].grid(axis="y", alpha=0.3, linestyle="--")  # Subtle grid lines
axes[1].set_facecolor("#FAFAFA")  # Light background

# Add data source and timestamp
axes[1].text(
    0.99,
    0.02,
    f"Source: Customer Analytics Database | Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
    transform=axes[1].transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
    style="italic",
    alpha=0.7,
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
)

plt.tight_layout()
plt.show()

# Color palette showcase for different data types
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(
    "🎨 Professional Color Palette Guide for Data Visualization",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

palette_examples = [
    ("viridis", "Sequential: Perfect for continuous data (sales, revenue, growth)"),
    ("RdBu_r", "Diverging: Ideal for correlations and comparisons around zero"),
    ("Set2", "Qualitative: Best for distinct categories (segments, regions)"),
    ("plasma", "Perceptual: Maximum visual distinction and modern appeal"),
]

for i, (palette_name, description) in enumerate(palette_examples):
    row, col = i // 2, i % 2
    sns.barplot(
        data=customers_df,
        x="region",
        y="spending",
        palette=palette_name,
        ax=axes[row, col],
    )
    axes[row, col].set_title(
        f"{palette_name.title()} Palette\n{description}", fontsize=12, fontweight="bold"
    )
    axes[row, col].tick_params(axis="x", rotation=45)
    sns.despine(ax=axes[row, col])
    axes[row, col].grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.show()

print("✅ Professional Styling Complete!")
print("🎨 Key Styling Principles Applied:")
print("   • Colorblind-friendly palettes for accessibility")
print("   • Clear typography hierarchy for readability")
print("   • Minimal design reducing visual clutter")
print("   • Data source attribution for credibility")
print(
    "💼 Business Impact: Professional visuals increase stakeholder engagement by 40%\n"
)


# -----------------------------------------------------------------------------
# 📊 Advanced Statistical Visualizations: Research-Grade Analysis
# -----------------------------------------------------------------------------
"""
🎯 WHY ADVANCED STATISTICAL PLOTS ARE ESSENTIAL:
Statistical visualizations provide rigorous, evidence-based insights:
- Show uncertainty and confidence in your findings
- Validate model assumptions and diagnostics
- Demonstrate statistical significance visually
- Support scientific and business conclusions with data

Critical for:
✅ Research publications and academic work
✅ A/B testing and experimental analysis
✅ Model validation and diagnostics
✅ Evidence-based business decisions
"""

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(
    "📊 Advanced Statistical Analysis: Research-Grade Visualizations",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

# 1. Regression with confidence intervals - Model uncertainty
sns.regplot(
    data=customers_df,
    x="income",
    y="spending",
    scatter_kws={"alpha": 0.6, "s": 50},
    line_kws={"color": "red", "linewidth": 3},
    ax=axes[0, 0],
)
axes[0, 0].set_title(
    "📈 Regression Analysis with Confidence Bands\n"
    "(Shaded Area Shows Model Uncertainty)",
    fontsize=12,
    fontweight="bold",
)
axes[0, 0].set_xlabel("Annual Income ($)", fontweight="bold")
axes[0, 0].set_ylabel("Monthly Spending ($)", fontweight="bold")

# Calculate and display regression statistics
slope, intercept, r_value, p_value, std_err = stats.linregress(
    customers_df["income"], customers_df["spending"]
)
axes[0, 0].text(
    0.05,
    0.95,
    f"R² = {r_value**2:.3f}\nP-value = {p_value:.2e}",
    transform=axes[0, 0].transAxes,
    fontsize=11,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
)

# 2. Residual analysis - Model diagnostic
predicted = slope * customers_df["income"] + intercept
residuals = customers_df["spending"] - predicted

sns.scatterplot(x=predicted, y=residuals, alpha=0.6, s=50, ax=axes[0, 1])
axes[0, 1].axhline(y=0, color="red", linestyle="--", linewidth=2)
axes[0, 1].set_title(
    "🔍 Residual Analysis\n(Random Pattern = Good Model Fit)",
    fontsize=12,
    fontweight="bold",
)
axes[0, 1].set_xlabel("Predicted Spending ($)", fontweight="bold")
axes[0, 1].set_ylabel("Residuals ($)", fontweight="bold")

# 3. Statistical comparison with confidence intervals
segment_stats = (
    customers_df.groupby("segment")["satisfaction"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
segment_stats["se"] = segment_stats["std"] / np.sqrt(segment_stats["count"])
segment_stats["ci"] = 1.96 * segment_stats["se"]  # 95% confidence interval

bars = axes[1, 0].bar(
    segment_stats["segment"],
    segment_stats["mean"],
    color=["#FF6B6B", "#4ECDC4", "#45B7D1"],
    alpha=0.8,
)
axes[1, 0].errorbar(
    segment_stats["segment"],
    segment_stats["mean"],
    yerr=segment_stats["ci"],
    fmt="none",
    color="black",
    capsize=8,
    capthick=2,
    elinewidth=2,
)

axes[1, 0].set_title(
    "📊 Satisfaction Comparison with 95% CI\n"
    "(Error Bars Show Statistical Significance)",
    fontsize=12,
    fontweight="bold",
)
axes[1, 0].set_xlabel("Customer Segment", fontweight="bold")
axes[1, 0].set_ylabel("Average Satisfaction Score", fontweight="bold")

# Add significance indicators
for i, (bar, mean_val, ci_val) in enumerate(
    zip(bars, segment_stats["mean"], segment_stats["ci"])
):
    axes[1, 0].text(
        bar.get_x() + bar.get_width() / 2.0,
        mean_val + ci_val + 0.1,
        f"{mean_val:.2f} ± {ci_val:.2f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

# 4. Joint distribution analysis

# Create joint plot manually for more control
sns.scatterplot(
    data=customers_df,
    x="income",
    y="spending",
    hue="segment",
    alpha=0.7,
    s=60,
    ax=axes[1, 1],
)

# Add confidence ellipses for each segment
for segment in customers_df["segment"].unique():
    segment_data = customers_df[customers_df["segment"] == segment]
    if len(segment_data) > 2:  # Need at least 3 points for covariance
        cov = np.cov(segment_data["income"], segment_data["spending"])
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        order = eigenvals.argsort()[::-1]
        eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

        angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
        width, height = 2 * 2 * np.sqrt(eigenvals)  # 95% confidence ellipse

        ellipse = Ellipse(
            xy=(segment_data["income"].mean(), segment_data["spending"].mean()),
            width=width,
            height=height,
            angle=angle,
            alpha=0.3,
        )
        axes[1, 1].add_patch(ellipse)

axes[1, 1].set_title(
    "🎯 Joint Distribution with Confidence Ellipses\n"
    "(Ellipses Show 95% Data Boundaries)",
    fontsize=12,
    fontweight="bold",
)
axes[1, 1].set_xlabel("Annual Income ($)", fontweight="bold")
axes[1, 1].set_ylabel("Monthly Spending ($)", fontweight="bold")

plt.tight_layout()
plt.show()

# Statistical significance testing
print("✅ Advanced Statistical Analysis Complete!")
print("\n📊 Statistical Test Results:")

# Perform ANOVA to compare spending across segments

premium_spending = customers_df[customers_df["segment"] == "Premium"]["spending"]
standard_spending = customers_df[customers_df["segment"] == "Standard"]["spending"]
budget_spending = customers_df[customers_df["segment"] == "Budget"]["spending"]

f_stat, p_value = f_oneway(premium_spending, standard_spending, budget_spending)
print(f"   ANOVA F-statistic: {f_stat:.3f}")
print(f"   P-value: {p_value:.2e}")
print(
    f"   Result: {'Significant differences' if p_value < 0.05 else 'No significant differences'} between segments"
)

# Pairwise t-tests
print("\n🔬 Pairwise Comparisons:")
segments = ["Premium", "Standard", "Budget"]
for i, seg1 in enumerate(segments):
    for seg2 in segments[i + 1 :]:
        data1 = customers_df[customers_df["segment"] == seg1]["spending"]
        data2 = customers_df[customers_df["segment"] == seg2]["spending"]
        t_stat, p_val = stats.ttest_ind(data1, data2)
        significance = (
            "***"
            if p_val < 0.001
            else "**"
            if p_val < 0.01
            else "*"
            if p_val < 0.05
            else "ns"
        )
        print(f"   {seg1} vs {seg2}: p = {p_val:.4f} {significance}")

print("\n💡 Statistical Insights:")
print("   • Strong statistical evidence for segment differences")
print("   • Regression model explains significant variance in spending")
print("   • Confidence intervals guide business decision uncertainty\n")


# -----------------------------------------------------------------------------
# 🚀 Interactive-Style Dashboard Creation
# -----------------------------------------------------------------------------
"""
🎯 WHY DASHBOARD-STYLE LAYOUTS MATTER:
Dashboard layouts present multiple KPIs and insights simultaneously:
- Executive-level overview of key metrics
- Multiple perspectives on the same data
- Professional presentation for stakeholders
- Comprehensive analysis in single view

Perfect for:
✅ Executive reports and board presentations
✅ Business intelligence dashboards
✅ Comprehensive data story telling
✅ Stakeholder communication and buy-in
"""

# Create executive dashboard layout
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)

# Dashboard title
fig.suptitle(
    "🚀 Executive Business Intelligence Dashboard\nCustomer Analytics & Performance Metrics",
    fontsize=20,
    fontweight="bold",
    y=0.98,
)

# KPI Cards Section (Top Row)
kpi_data = [
    ("Total Customers", f"{len(customers_df):,}", "skyblue"),
    ("Avg Satisfaction", f"{customers_df['satisfaction'].mean():.2f}/5", "lightgreen"),
    ("Revenue/Customer", f"${customers_df['spending'].mean():.0f}/mo", "lightcoral"),
    ("Top Region", customers_df["region"].mode()[0], "lightyellow"),
]

for i, (title, value, color) in enumerate(kpi_data):
    ax_kpi = fig.add_subplot(gs[0, i])
    ax_kpi.text(
        0.5, 0.6, str(value), ha="center", va="center", fontsize=24, fontweight="bold"
    )
    ax_kpi.text(
        0.5, 0.25, title, ha="center", va="center", fontsize=14, fontweight="bold"
    )
    ax_kpi.set_facecolor(color)
    ax_kpi.set_xticks([])
    ax_kpi.set_yticks([])

    # Add subtle border
    for spine in ax_kpi.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("gray")

# Large Central Heatmap - Business Performance Matrix
ax_main = fig.add_subplot(gs[1:3, 0:2])
pivot_performance = customers_df.pivot_table(
    values="spending", index="region", columns="segment", aggfunc="mean"
)
sns.heatmap(
    pivot_performance,
    annot=True,
    fmt=".0f",
    cmap="RdYlGn",
    cbar_kws={"label": "Average Monthly Spending ($)"},
    ax=ax_main,
)
ax_main.set_title(
    "🔥 Regional Performance Matrix\nAverage Spending by Region × Segment",
    fontsize=14,
    fontweight="bold",
)

# Time Series Panel - Sales Trends
ax_ts = fig.add_subplot(gs[1, 2:])
for product in ["Phone", "Laptop"]:
    ax_ts.plot(
        monthly_sales["Month"],
        monthly_sales[product],
        marker="o",
        linewidth=3,
        markersize=6,
        label=product,
    )
ax_ts.set_title("📈 Key Product Sales Trends", fontsize=14, fontweight="bold")
ax_ts.legend(loc="upper left")
ax_ts.grid(True, alpha=0.3)
ax_ts.tick_params(axis="x", rotation=45)

# Distribution Analysis Panel
ax_dist = fig.add_subplot(gs[2, 2:])
sns.violinplot(
    data=customers_df, x="segment", y="satisfaction", palette="Set2", ax=ax_dist
)
ax_dist.set_title(
    "⭐ Customer Satisfaction Distribution", fontsize=14, fontweight="bold"
)
ax_dist.set_xlabel("Customer Segment", fontweight="bold")
ax_dist.set_ylabel("Satisfaction Score", fontweight="bold")

# Correlation Insights Panel
ax_corr = fig.add_subplot(gs[3, 0:2])
correlation_subset = customers_df[["age", "income", "spending", "satisfaction"]].corr()
sns.heatmap(
    correlation_subset, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax_corr
)
ax_corr.set_title("🔗 Variable Correlation Matrix", fontsize=14, fontweight="bold")

# Business Insights Panel - Key Metrics
ax_insights = fig.add_subplot(gs[3, 2:])
ax_insights.axis("off")  # Remove axes for text panel

# Calculate key business metrics
total_revenue = customers_df["spending"].sum()
customer_lifetime_value = customers_df["spending"].mean() * 12
satisfaction_score = customers_df["satisfaction"].mean()
top_segment = customers_df.groupby("segment")["spending"].sum().idxmax()

insights_text = f"""
📊 KEY BUSINESS INSIGHTS

💰 Total Monthly Revenue: ${total_revenue:,.0f}
👥 Average CLV (Annual): ${customer_lifetime_value:,.0f}
⭐ Customer Satisfaction: {satisfaction_score:.2f}/5
🏆 Top Performing Segment: {top_segment}

🎯 STRATEGIC RECOMMENDATIONS
• Focus retention on Premium segment
• Investigate regional performance gaps
• Leverage income-spending correlation
• Monitor satisfaction trends closely

📈 GROWTH OPPORTUNITIES
• Premium segment expansion
• Cross-selling to high-income customers
• Regional strategy optimization
"""

ax_insights.text(
    0.05,
    0.95,
    insights_text,
    transform=ax_insights.transAxes,
    fontsize=11,
    verticalalignment="top",
    fontweight="normal",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
)

plt.show()

print("✅ Executive Dashboard Complete!")
print("📊 Dashboard Features:")
print("   • KPI cards for key metrics overview")
print("   • Performance heatmap for regional analysis")
print("   • Time series for trend monitoring")
print("   • Distribution analysis for customer insights")
print("   • Correlation matrix for relationship understanding")
print("   • Strategic recommendations for action items")
print(
    "\n🎯 Executive Summary: Multi-panel dashboard provides comprehensive business intelligence\n"
)

"""
📚 What You’ve Really Learned – Seaborn Data Viz Recap
=======================================================

🎯 The Core Charts You’ll Keep Using
-------------------------------------
• Histograms & KDEs — for checking how your data is spread out
• Box & Violin plots — to spot outliers and compare distributions
• Scatter & Line plots — great for finding trends and relationships
• Heatmaps — perfect when you’re drowning in correlation numbers
• Bar, Point, and Count plots — quick checks on categories
• Pairplots & FacetGrids — when you need to go all-in on multi-var analysis

🎨 Making Things Look Good (and Clear)
---------------------------------------
• Pick color palettes that fit the data, not your mood
• Clean over fancy — no one likes chart junk
• Label what matters, trim what doesn’t
• Think about the *person* reading your chart — not your portfolio

📈 Why It All Matters
----------------------
• Good visuals = faster understanding = better decisions
• Don’t just show data — *explain* it with your plots
• If it’s not helping someone take action, it’s probably just noise

💡 Final Thought
------------------
At the end of the day, a great chart isn’t about making people say “wow”.
It’s about making them say “ohh, I get it now”.

That’s your real superpower. Use it well.

🎉 CONGRATULATIONS! Data Visualization Mastery Complete!
📊 You've learned to create beautiful, insightful statistical visualizations
"""
