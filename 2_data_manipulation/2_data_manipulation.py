"""
📊 Data Manipulation with NumPy & Pandas
===========================================

This tutorial is your friendly intro to wrangling and working with structured data in Python.
From cleaning up messy CSVs to slicing, grouping, and reshaping — you’ll get hands-on with the tools that power real data analysis.

🎯 What you'll learn:
- NumPy arrays for fast, efficient number crunching
- Pandas DataFrames for organizing tabular data
- How to clean up messy datasets (nulls, typos, weird formats)
- Grouping and aggregation for quick summaries
- Merging and joining multiple datasets
- Reshaping tricks using melt(), pivot(), and more
- Time series basics: dates, times, and trends

🔧 How to use:
Everything runs top to bottom — no setup headaches.
Just follow the flow, and check the printed output as you go.
Each section has clear code and comments to guide you.

📦 Requirements:
You'll need just the usual suspects: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, and SciPy.
Install them with:

    pip install -r requirements.txt

🤖 Note:
This is part of the [Cheatsheet DS Minicourse] — a side project made with
some coding, some love, and some help from Claude AI.

Let’s get started! 🧠
"""

import numpy as np
import pandas as pd
import warnings
import time

# Optional: Makes DataFrame outputs easier to read in terminal
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 150)

# Suppress warnings for cleaner output in tutorial
warnings.filterwarnings("ignore")

print("🎉 Welcome to the Complete Data Manipulation Tutorial!")
print("=" * 55)
print("📚 Learn by doing - every concept includes working examples!")
print()

# ==============================================================================
# 📊 SECTION 1: NumPy Fundamentals - The Mathematical Foundation
# ==============================================================================

"""
🧠 Why NumPy Matters:
NumPy is the foundation of Python's data science ecosystem. It provides:
- Homogeneous arrays that are 50-100x faster than Python lists
- Vectorized operations (no manual loops needed)
- Broadcasting for operations between different-sized arrays
- Memory-efficient storage of large datasets

Think of NumPy as your mathematical calculator on steroids!
"""

print("📊 SECTION 1: NUMPY FUNDAMENTALS")
print("=" * 35)
print("🎯 Why NumPy? Speed and efficiency for numerical computations!")
print()

# 🏗️ Creating NumPy Arrays - The Building Blocks
print("🏗️ Creating Arrays:")

# From Python lists (most common way to start)
numbers_1d = np.array([1, 2, 3, 4, 5])  # 1D array - like a column in Excel
matrix_2d = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array - like a spreadsheet table
cube_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3D array - stacked tables

print(f"1D array: {numbers_1d}")
print(f"2D matrix:\n{matrix_2d}")
print(
    f"Shape tells us dimensions: {matrix_2d.shape} = {matrix_2d.shape[0]} rows × {matrix_2d.shape[1]} columns"
)
print()

# 🎯 Special array creation functions - very useful for initializing data
zeros_array = np.zeros((3, 4))  # Initialize with zeros (common for placeholders)
ones_array = np.ones((2, 3))  # Initialize with ones (useful for calculations)
range_array = np.arange(0, 10, 2)  # Like Python's range() but returns array
evenly_spaced = np.linspace(0, 5, 6)  # 6 equally spaced numbers from 0 to 5
random_array = np.random.random((2, 3))  # Random numbers between 0 and 1

print(f"Zeros (3×4): \n{zeros_array}")
print(f"Range 0-10 step 2: {range_array}")
print(f"6 evenly spaced 0-5: {evenly_spaced}")
print()

# 🚀 Vectorized Operations - NumPy's Superpower
print("🚀 Vectorized Operations (No Loops Required!):")

array_a = np.array([1, 2, 3, 4])
array_b = np.array([10, 20, 30, 40])

# Element-wise operations happen automatically - much faster than Python loops
print(f"Array A: {array_a}")
print(f"Array B: {array_b}")
print(f"A + B: {array_a + array_b}")  # Each element: A[i] + B[i]
print(
    f"A * B: {array_a * array_b}"
)  # Element-wise multiplication (not matrix multiplication)
print(f"A squared: {array_a**2}")  # Apply exponent to each element
print(f"Square root of A: {np.sqrt(array_a)}")  # Built-in mathematical functions
print()

# 📊 Statistical Operations - Get insights from your data instantly
print("📊 Built-in Statistical Functions:")
sample_data = np.array([85, 90, 78, 92, 88, 76, 95, 89, 91, 87])  # Test scores example

print(f"Test scores: {sample_data}")
print(f"Mean (average): {np.mean(sample_data):.1f}")
print(f"Median (middle value): {np.median(sample_data):.1f}")
print(f"Standard deviation: {np.std(sample_data):.1f}")
print(f"Min score: {np.min(sample_data)}, Max score: {np.max(sample_data)}")
print()

# 🎯 Boolean Indexing - Filter Data Based on Conditions
print("🎯 Boolean Indexing - Smart Data Filtering:")

# Set seed for reproducible random numbers (important for tutorials!)
np.random.seed(42)
sample_data = np.random.randint(1, 20, 10)  # 10 random integers from 1-19

print(f"Original data: {sample_data}")

# Create boolean mask - array of True/False values
mask_above_10 = sample_data > 10  # Returns boolean array
print(f"Values > 10? {mask_above_10}")  # Shows True/False for each element

# Use mask to filter data - only keeps True positions
filtered_data = sample_data[mask_above_10]
print(f"Filtered values (>10): {filtered_data}")

# Multiple conditions - use & (and) or | (or), wrap each condition in parentheses
complex_mask = (sample_data > 5) & (sample_data < 15)
print(f"Values between 5-15: {sample_data[complex_mask]}")
print()

"""
🎯 NumPy Recap:
- Arrays are faster and more memory-efficient than Python lists
- Vectorized operations eliminate the need for manual loops
- Boolean indexing provides powerful data filtering capabilities
- Built-in statistical functions make data analysis effortless
"""

# ==============================================================================
# 📊 SECTION 2: Pandas DataFrames - Your Data Analysis Workspace
# ==============================================================================

"""
🎯 Why Pandas DataFrames Matter:
DataFrames are like Excel spreadsheets with superpowers. They can:
- Handle mixed data types (numbers, text, dates) in a single structure
- Provide labeled rows and columns for intuitive data access
- Offer powerful data manipulation and analysis tools
- Handle missing data gracefully
- Scale to millions of records efficiently

Think of DataFrames as your primary tool for real-world data analysis!
"""

print("📊 SECTION 2: PANDAS DATAFRAMES - YOUR DATA WORKSPACE")
print("=" * 55)
print("🎯 Why DataFrames? Structure + flexibility for real-world data!")
print()

# 🎲 Set random seed for reproducible results
np.random.seed(42)

# 🏗️ Creating Realistic Business Dataset
print("🏗️ Creating a Realistic Business Dataset:")

# Real-world data often comes from multiple sources - here we simulate a sales dataset
# Each dictionary key becomes a column name, values become column data
sales_data = {
    # pd.date_range creates consecutive dates - common in time series analysis
    "date": pd.date_range("2024-01-01", periods=15, freq="D"),
    "product": np.random.choice(["Laptop", "Mouse", "Keyboard", "Monitor"], 15),
    "category": np.random.choice(["Electronics", "Accessories"], 15),
    "price": np.random.choice([999, 25, 75, 300], 15),  # Realistic product prices
    "quantity": np.random.randint(1, 6, 15),  # 1-5 items per sale
    "customer_rating": np.round(
        np.random.uniform(3.0, 5.0, 15), 1
    ),  # Ratings from 3.0-5.0
    "sales_rep": np.random.choice(["Alice", "Bob", "Charlie"], 15),
}

# Create DataFrame from dictionary - this is the most common way to start
df = pd.DataFrame(sales_data)

# Add calculated column - very common in business analysis
df["total_sales"] = df["price"] * df["quantity"]  # Revenue = Price × Quantity

print("📋 Our sample business dataset:")
print(df.head())  # .head() shows first 5 rows by default
print()

# 🔍 Essential DataFrame Inspection - Always Start Here!
print("🔍 Essential Dataset Inspection:")
print("💡 Always explore your data before analysis!")

print(f"📏 Dataset dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"📋 Column names: {list(df.columns)}")
print()

print("📊 Data types (very important for analysis):")
print(df.dtypes)
print()

print("📈 Statistical summary of numerical columns:")
print(df.describe())
print()

print("🔍 Memory usage (important for large datasets):")
memory_usage = df.memory_usage(deep=True).sum()
print(f"Total memory: {memory_usage:,} bytes ({memory_usage / 1024:.1f} KB)")
print()

"""
🎯 DataFrame Basics Recap:
- DataFrames combine the structure of spreadsheets with programming power
- Always inspect new data with .head(), .shape, .dtypes, .describe()
- Mixed data types (numbers, text, dates) work seamlessly together
- Memory efficiency matters as datasets grow larger
"""

# ==============================================================================
# 🔍 SECTION 3: Data Selection & Indexing - Finding Your Data
# ==============================================================================

"""
🎯 Why Data Selection Matters:
Most data analysis involves working with subsets of your data. You need to:
- Select specific columns for focused analysis
- Filter rows based on business conditions
- Extract data ranges for time series analysis
- Combine multiple selection criteria for complex queries

Mastering selection is essential for efficient data analysis!
"""

print("🔍 SECTION 3: DATA SELECTION & INDEXING")
print("=" * 40)
print("🎯 Why Selection? Focus on the data that matters for your analysis!")
print()

# 📋 Column Selection - Focus on Relevant Data
print("📋 Column Selection Techniques:")

# Single column returns a Series (1D structure)
single_column = df["product"]
print(f"Single column type: {type(single_column).__name__}")
print(f"First few products: {single_column.head(3).values}")

# Multiple columns return a DataFrame (2D structure) - note the double brackets!
multiple_columns = df[["product", "price", "total_sales"]]
print(f"Multiple columns type: {type(multiple_columns).__name__}")
print("Selected columns:")
print(multiple_columns.head(3))
print()

# 🎯 Boolean Filtering - The Heart of Data Analysis
print("🎯 Boolean Filtering - Find Data That Meets Your Criteria:")

# Single condition - find high-value sales
high_value_mask = df["total_sales"] > 500
high_value_sales = df[high_value_mask]  # Can also write as df[df['total_sales'] > 500]
print(f"💰 High-value sales (>$500): {len(high_value_sales)} out of {len(df)} records")
print(high_value_sales[["product", "total_sales", "customer_rating"]].head(3))
print()

# Multiple conditions - CRITICAL: Use & for AND, | for OR (not 'and'/'or' keywords!)
# Each condition must be wrapped in parentheses
electronics_highly_rated = df[
    (df["category"] == "Electronics") & (df["customer_rating"] >= 4.5)
]
print(f"⭐ High-rated electronics: {len(electronics_highly_rated)} records")
if len(electronics_highly_rated) > 0:
    print(
        electronics_highly_rated[["product", "customer_rating", "total_sales"]].head()
    )
print()

# 🎪 .isin() Method - Check Multiple Values at Once
print("🎪 Multiple Value Filtering with .isin():")

# Like SQL's IN operator - very handy for category filtering
premium_products = df[df["product"].isin(["Laptop", "Monitor"])]
print(f"💻 Premium products (Laptop/Monitor): {len(premium_products)} records")
print(f"Premium product sales: {premium_products['total_sales'].sum():.2f}")
print()

# 🔧 .loc vs .iloc - Label vs Position Based Selection
print("🔧 .loc vs .iloc - Understanding the Difference:")
print("💡 .loc uses labels (column names, row conditions)")
print("💡 .iloc uses integer positions (like array indexing)")

# .loc - Label-based selection (most common in data analysis)
# Select rows where date >= '2024-01-10', show only specific columns
recent_sales = df.loc[df["date"] >= "2024-01-10", ["product", "total_sales", "date"]]
print("\n📅 .loc example - Recent sales (from Jan 10):")
print(recent_sales.head(3))

# .iloc - Integer position-based selection (useful for systematic sampling)
# First 3 rows, first 4 columns (like array slicing)
first_subset = df.iloc[:3, :4]
print("\n📊 .iloc example - First 3 rows, first 4 columns:")
print(first_subset)

# Advanced .loc usage - combine row and column selection
high_value_summary = df.loc[
    df["total_sales"] > 200, ["product", "price", "quantity", "total_sales"]
]
print(f"\n💰 High-value transactions summary ({len(high_value_summary)} records):")
print(high_value_summary.head())
print()

"""
🎯 Selection & Indexing Recap:
- Single brackets [column] return Series, double brackets [[columns]] return DataFrame
- Boolean masks filter rows based on conditions
- Use & (and), | (or) for multiple conditions, wrap each in parentheses
- .loc uses labels and conditions (preferred for data analysis)
- .iloc uses integer positions (useful for systematic selection)
- .isin() efficiently checks membership in a list of values
"""

# ==============================================================================
# 🧹 SECTION 4: Data Cleaning - Handling Real-World Messiness
# ==============================================================================

"""
🎯 Why Data Cleaning Is Crucial:
Real-world data is messy! Common problems include:
- Missing values (blank cells, NaN, None)
- Inconsistent formatting (mixed case, extra spaces)
- Invalid data types (numbers stored as text)
- Inconsistent categorical values
- Outliers and erroneous entries

Data scientists spend 80% of their time cleaning data - master these skills!
"""

print("🧹 SECTION 4: DATA CLEANING - REAL-WORLD MESSINESS")
print("=" * 50)
print("🎯 Why Cleaning? Real data is messy - make it analysis-ready!")
print()

# 🎭 Create Intentionally Messy Data (Realistic Scenario)
print("🎭 Creating Realistic Messy Data:")

messy_data = {
    "name": ["John Doe", "jane smith", "BOB JOHNSON", None, "Alice Brown", ""],
    "email": [
        "john@email.com",
        "JANE@EMAIL.COM",
        "bob@invalid",
        "alice@email.com",
        None,
        "eve@email.com",
    ],
    "age": [25, 30, None, 35, 28, 45],
    "salary": [
        "50000",
        "60,000",
        "75000",
        None,
        "55,000",
        "$80000",
    ],  # Mixed formats - very common!
    "department": ["Sales", "sales", "MARKETING", "Marketing", "HR", "hr"],
}

messy_df = pd.DataFrame(messy_data)
print("🚨 Original messy data:")
print(messy_df)
print()

# Check for missing values - critical first step
print("🔍 Missing values per column:")
missing_counts = messy_df.isnull().sum()
print(missing_counts)
print(f"Total missing values: {messy_df.isnull().sum().sum()}")
print()

# 🔧 Data Cleaning Pipeline - Always Work on a Copy!
print("🔧 Data Cleaning Pipeline:")
print("💡 Always work on a copy to preserve original data!")

clean_df = messy_df.copy()  # Preserve original data

# Step 1: Clean name column
print("\n📝 Step 1: Cleaning name column...")
# Empty string '' is different from None/NaN - normalize them
clean_df["name"] = clean_df["name"].replace("", None)  # Empty string → NaN
clean_df["name"] = clean_df["name"].fillna("Unknown")  # NaN → 'Unknown'
clean_df["name"] = clean_df["name"].str.title()  # Standardize to Title Case
print("Names after cleaning:", clean_df["name"].tolist())

# Step 2: Clean email addresses
print("\n📧 Step 2: Cleaning email addresses...")
# .str accessor applies string methods to entire column (vectorized string operations)
clean_df["email"] = clean_df["email"].str.lower()  # Standardize to lowercase

# Email validation using regex pattern matching
email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
clean_df["valid_email"] = clean_df["email"].str.match(email_pattern, na=False)
print("Email validation results:")
print(clean_df[["email", "valid_email"]])

# Step 3: Clean salary column (Very Common Challenge!)
print("\n💰 Step 3: Cleaning salary column...")
print("Original salary values:", messy_df["salary"].tolist())

# Remove currency symbols and thousands separators, then convert to numeric
clean_df["salary"] = (
    clean_df["salary"]
    .astype(str)  # Ensure all are strings first
    .str.replace("$", "", regex=False)  # Remove dollar signs
    .str.replace(",", "", regex=False)
)  # Remove commas

# Convert to numeric, coercing errors to NaN
clean_df["salary"] = pd.to_numeric(clean_df["salary"], errors="coerce")

# Fill missing salaries with median (more robust than mean for outliers)
median_salary = clean_df["salary"].median()
clean_df["salary"] = clean_df["salary"].fillna(median_salary)
print(f"Cleaned salaries: {clean_df['salary'].tolist()}")
print(f"Median salary used for missing values: ${median_salary:,.0f}")

# Step 4: Standardize department names
print("\n🏢 Step 4: Standardizing department names...")
# Create mapping for inconsistent categorical data
dept_mapping = {
    "sales": "Sales",
    "Sales": "Sales",
    "MARKETING": "Marketing",
    "Marketing": "Marketing",
    "hr": "HR",
    "HR": "HR",
}
clean_df["department"] = clean_df["department"].map(dept_mapping)
print("Standardized departments:", clean_df["department"].unique())
print()

print("✅ Final cleaned data:")
print(clean_df)
print()

# 📊 Before/After Comparison
print("📊 Cleaning Results Summary:")
print(f"Missing values before cleaning: {messy_df.isnull().sum().sum()}")
print(f"Missing values after cleaning: {clean_df.isnull().sum().sum()}")
print(f"Invalid emails identified: {(~clean_df['valid_email']).sum()}")
print(f"Salary values standardized: {(clean_df['salary'] > 0).sum()}/{len(clean_df)}")
print()

"""
🎯 Data Cleaning Recap:
- Always work on a copy to preserve original data
- Handle missing values systematically (fillna, replace, drop)
- Standardize text data (case, format, categorical mappings)
- Convert data types appropriately (to_numeric, astype)
- Validate data quality with patterns (regex for emails, ranges for numbers)
- Document cleaning steps for reproducibility
"""

# ==============================================================================
# 📝 SECTION 5: String Operations - Text Data Mastery
# ==============================================================================

"""
🎯 Why String Operations Matter:
Text data is everywhere in business:
- Customer feedback and reviews
- Product descriptions and categories
- Survey responses and comments
- Names, addresses, and contact information
- Social media content and web data

Pandas string operations help you extract insights from unstructured text!
"""

print("📝 SECTION 5: STRING OPERATIONS - TEXT DATA MASTERY")
print("=" * 50)
print("🎯 Why Strings? Extract insights from text data!")
print()

# 🎭 Create Sample Customer Review Data
print("🎭 Sample Customer Review Analysis:")

reviews_df = pd.DataFrame(
    {
        "review_text": [
            "Great product! Highly recommended.",
            "Not bad, could be better... 3/5",
            "AMAZING!!! Will buy again!!!",
            "Poor quality. Waste of money.",
            "Good value for money. Fast shipping.",
            "Excellent customer service!",
            "average product, nothing special",
            "TERRIBLE experience. Very disappointed.",
        ],
        "product": [
            "Laptop",
            "Mouse",
            "Keyboard",
            "Monitor",
            "Tablet",
            "Phone",
            "Headphones",
            "Speaker",
        ],
        "review_date": pd.date_range("2024-01-01", periods=8, freq="3D"),
    }
)

print("📋 Customer reviews dataset:")
print(reviews_df[["product", "review_text"]])
print()

# 📊 Text Analysis with .str Accessor
print("📊 Text Analysis using .str accessor:")
print("💡 .str allows you to apply string methods to entire columns!")

"""
# -------------------------------------------------------------
# 🔤 String Filtering with str.contains()
# -------------------------------------------------------------

# Safely filter rows where 'Department' column mentions "Data"
# na=False prevents errors if there are missing values
filtered_df = df[df["Department"].str.contains("Data", case=False, na=False)]
print("\n📄 Filtered Rows where Department contains 'Data':")
print(filtered_df)

"""

# Basic text metrics
reviews_df["char_count"] = reviews_df["review_text"].str.len()
reviews_df["word_count"] = reviews_df["review_text"].str.split().str.len()
reviews_df["exclamation_count"] = reviews_df["review_text"].str.count("!")

print("Basic text metrics:")
text_metrics = reviews_df[["product", "char_count", "word_count", "exclamation_count"]]
print(text_metrics)
print()

# 🎯 Pattern Detection and Extraction
print("🎯 Pattern Detection and Text Extraction:")

# Detect ALL CAPS text (often indicates strong emotions)
reviews_df["has_caps"] = reviews_df["review_text"].str.contains("[A-Z]{3,}")

# Extract sentiment words using regex
sentiment_pattern = r"(great|amazing|good|excellent|poor|bad|terrible|awful)"
reviews_df["sentiment_word"] = (
    reviews_df["review_text"]
    .str.lower()  # Convert to lowercase first
    .str.extract(sentiment_pattern, expand=False)
)

# Find reviews with ratings (numbers followed by /5)
reviews_df["has_rating"] = reviews_df["review_text"].str.contains(r"\d+/5")
reviews_df["extracted_rating"] = reviews_df["review_text"].str.extract(
    r"(\d+)/5", expand=False
)

print("Pattern detection results:")
pattern_results = reviews_df[
    ["product", "has_caps", "sentiment_word", "extracted_rating"]
]
print(pattern_results)
print()

# 🔧 Text Cleaning and Standardization
print("🔧 Text Cleaning Operations:")

# Create cleaned version of review text
reviews_df["review_clean"] = (
    reviews_df["review_text"]
    .str.lower()  # Standardize case
    .str.replace("[^\w\s]", " ", regex=True)  # Remove punctuation
    .str.replace("\s+", " ", regex=True)  # Multiple spaces → single space
    .str.strip()
)  # Remove leading/trailing whitespace

print("Before and after text cleaning:")
cleaning_comparison = reviews_df[["review_text", "review_clean"]].head(4)
for idx, row in cleaning_comparison.iterrows():
    print(f"Original: {row['review_text']}")
    print(f"Cleaned:  {row['review_clean']}")
    print()

# 📊 Sentiment Analysis Summary
print("📊 Text Analysis Summary:")
sentiment_summary = reviews_df.groupby("sentiment_word").size()
print("Sentiment word frequency:")
print(sentiment_summary)

caps_analysis = reviews_df["has_caps"].value_counts()
print(f"\nReviews with ALL CAPS: {caps_analysis.get(True, 0)} out of {len(reviews_df)}")

avg_length_by_sentiment = reviews_df.groupby("sentiment_word")["word_count"].mean()
print("\nAverage review length by sentiment:")
print(avg_length_by_sentiment.round(1))
print()

"""
🎯 String Operations Recap:
- .str accessor applies string methods to entire columns (vectorized)
- Common operations: len(), split(), contains(), extract(), replace()
- Regular expressions (regex) enable powerful pattern matching
- Text cleaning involves case standardization, punctuation removal, whitespace handling
- String analysis reveals insights about sentiment, engagement, and content quality
"""

# ==============================================================================
# 📊 SECTION 6: GroupBy & Aggregation - Business Intelligence
# ==============================================================================

"""
🎯 Why GroupBy Is Essential:
GroupBy operations answer critical business questions:
- "What are our sales by product category?"
- "Which sales rep is performing best?"
- "How do customer ratings vary by product?"
- "What's our monthly revenue trend?"

GroupBy splits data into groups, applies functions, and combines results -
it's like creating pivot tables programmatically!
"""

print("📊 SECTION 6: GROUPBY & AGGREGATION - BUSINESS INTELLIGENCE")
print("=" * 60)
print("🎯 Why GroupBy? Answer business questions with grouped analysis!")
print()

# Return to our sales dataset for business analysis
print("📈 Business Analysis of Our Sales Data:")
print(f"Total records to analyze: {len(df)}")
print()

# 📊 Dataset-Level KPIs (Key Performance Indicators)
print("🏢 Overall Business KPIs:")

total_revenue = df["total_sales"].sum()
avg_transaction = df["total_sales"].mean()
avg_rating = df["customer_rating"].mean()
unique_products = df["product"].nunique()
total_transactions = len(df)

print(f"💰 Total Revenue: ${total_revenue:,.2f}")
print(f"💳 Average Transaction: ${avg_transaction:.2f}")
print(f"⭐ Average Rating: {avg_rating:.2f}/5.0")
print(f"📦 Unique Products: {unique_products}")
print(f"📊 Total Transactions: {total_transactions}")
print()

# 🎯 Single-Column GroupBy - Basic Business Insights
print("🎯 Sales Performance by Product:")

# .groupby() splits data by product, .agg() applies multiple functions
product_performance = (
    df.groupby("product")
    .agg(
        {
            "total_sales": ["sum", "mean", "count"],  # Revenue metrics
            "customer_rating": "mean",  # Satisfaction metric
            "quantity": "sum",  # Volume metric
        }
    )
    .round(2)
)

# Flatten multi-level column names for easier reading
product_performance.columns = [
    "revenue_total",
    "revenue_avg",
    "transaction_count",
    "avg_rating",
    "units_sold",
]

# Sort by total revenue (descending) to see best performers first
product_performance = product_performance.sort_values("revenue_total", ascending=False)

print(product_performance)
print()

# 📊 Multi-Column GroupBy - Deeper Business Insights
print("📊 Revenue Analysis by Category and Sales Rep:")

# Group by two columns creates hierarchical analysis
category_rep_analysis = (
    df.groupby(["category", "sales_rep"])
    .agg({"total_sales": ["sum", "count"], "customer_rating": "mean"})
    .round(2)
)

# Flatten columns and rename for clarity
category_rep_analysis.columns = ["total_revenue", "transaction_count", "avg_rating"]
print(category_rep_analysis)
print()

# 🎯 Custom Aggregation Functions - Specialized Business Metrics
print("🎯 Custom Business Metrics:")


def sales_volatility(series):
    """Calculate coefficient of variation - measures sales consistency"""
    if series.std() == 0 or series.mean() == 0:
        return 0
    return (series.std() / series.mean()) * 100


def hit_rate(series):
    """Calculate percentage of high-value transactions (>$300)"""
    return (series > 300).mean() * 100


# Apply custom functions alongside built-in ones
custom_metrics = (
    df.groupby("product")
    .agg(
        {
            "total_sales": ["min", "max", sales_volatility, hit_rate],
            "customer_rating": ["min", "max"],
        }
    )
    .round(2)
)

custom_metrics.columns = [
    "min_sale",
    "max_sale",
    "volatility_%",
    "hit_rate_%",
    "min_rating",
    "max_rating",
]
print("Sales volatility and hit rates by product:")
print(custom_metrics)
print()

# 📈 Time-Based GroupBy Analysis
print("📈 Daily Sales Trend Analysis:")

# Group by date to see daily performance
daily_performance = (
    df.groupby("date")
    .agg({"total_sales": "sum", "quantity": "sum", "customer_rating": "mean"})
    .round(2)
)

daily_performance.columns = ["daily_revenue", "daily_units", "daily_avg_rating"]

print("First week of sales performance:")
print(daily_performance.head(7))
print()

# 🎯 Advanced GroupBy Techniques
print("🎯 Advanced GroupBy Techniques:")

# Multiple aggregations with different functions per column
advanced_analysis = (
    df.groupby("sales_rep")
    .agg(
        {
            "total_sales": ["sum", "mean", "std"],
            "customer_rating": ["mean", "min", "max"],
            "product": "nunique",  # Count unique products sold
        }
    )
    .round(2)
)

# Rename columns for clarity
advanced_analysis.columns = [
    "total_revenue",
    "avg_transaction",
    "sales_std",
    "avg_rating",
    "min_rating",
    "max_rating",
    "products_sold",
]

print("Comprehensive sales rep performance:")
print(advanced_analysis)
print()

"""
🎯 GroupBy & Aggregation Recap:
- GroupBy splits data into groups based on column values
- Aggregation functions (sum, mean, count, etc.) summarize each group
- Multi-column grouping creates hierarchical analysis
- Custom functions enable specialized business metrics
- .agg() allows multiple functions per column in one operation
- Results can be sorted and filtered like regular DataFrames
"""

# ==============================================================================
# 🔄 SECTION 7: Reshaping Data - Pivot Tables & Melting
# ==============================================================================

"""
🎯 Why Data Reshaping Matters:
Different analysis tasks require different data formats:
- Wide format: Each variable has its own column (good for analysis)
- Long format: Variables are stacked in rows (good for visualization)
- Pivot tables: Cross-tabulation for business reporting
- Melting: Convert wide data to long format

Reshaping is essential for data visualization and advanced analysis!
"""

print("🔄 SECTION 7: RESHAPING DATA - PIVOT TABLES & MELTING")
print("=" * 55)
print("🎯 Why Reshape? Transform data structure to match your analysis needs!")
print()

# 🎯 Pivot Tables - Cross-Tabulation Analysis
print("🎯 Pivot Tables - Business Cross-Tabulation:")
print("💡 Pivot tables answer: 'How much did X sell in Y category?'")

# Create cross-tabulation: products (rows) vs sales_rep (columns)
pivot_sales = df.pivot_table(
    values="total_sales",  # What to aggregate
    index="product",  # Rows (what goes down)
    columns="sales_rep",  # Columns (what goes across)
    aggfunc="sum",  # How to aggregate (sum, mean, count, etc.)
    fill_value=0,  # Replace NaN with 0
    margins=True,  # Add row/column totals ('All' row/column)
)

print("💹 Sales by Product × Sales Rep:")
print(pivot_sales)
print()

# Multi-metric pivot table
pivot_detailed = df.pivot_table(
    values=["total_sales", "customer_rating"],  # Multiple metrics
    index="product",
    columns="category",
    aggfunc={
        "total_sales": "sum",
        "customer_rating": "mean",
    },  # Different functions per metric
    fill_value=0,
).round(2)

print("📊 Multi-Metric Pivot: Sales & Ratings by Product × Category:")
print(pivot_detailed)
print()

# 🔄 Melting - Wide to Long Format Transformation
print("🔄 Melting Data - Wide to Long Format:")
print("💡 Melting stacks columns into rows for visualization and analysis")

# Create sample wide-format data (common in Excel exports)
wide_data = pd.DataFrame(
    {
        "product": ["Laptop", "Mouse", "Keyboard"],
        "Q1_sales": [15000, 2500, 1800],
        "Q2_sales": [18000, 2200, 2100],
        "Q3_sales": [16500, 2800, 1900],
        "Q4_sales": [17200, 2400, 2000],
    }
)

print("📊 Original wide-format data (typical Excel export):")
print(wide_data)
print()

# Melt to long format - quarters become a single column
melted_data = pd.melt(
    wide_data,
    id_vars=["product"],  # Columns to keep as identifiers
    value_vars=["Q1_sales", "Q2_sales", "Q3_sales", "Q4_sales"],  # Columns to melt
    var_name="quarter",  # Name for the new column containing old column names
    value_name="sales",  # Name for the new column containing the values
)

# Clean up the quarter column
melted_data["quarter"] = melted_data["quarter"].str.replace("_sales", "")

print("📈 Melted to long format (better for analysis and plotting):")
print(melted_data)
print()

# 🔧 Pivot vs Melt - When to Use Each
print("🔧 Pivot vs Melt - Choosing the Right Tool:")

# Example: Melt our original sales data for time series analysis
sales_melted = pd.melt(
    df,
    id_vars=["date", "product", "sales_rep"],
    value_vars=["price", "quantity", "total_sales"],
    var_name="metric",
    value_name="value",
)

print("🔍 Melted sales data (first 10 rows):")
print(sales_melted.head(10))
print()

# Now we can easily analyze all metrics together
metric_summary = sales_melted.groupby(["metric", "product"])["value"].mean().round(2)
print("📊 Average values by metric and product:")
print(metric_summary.head(12))  # Show first 12 entries
print()

# 🌟 Advanced Reshaping - Stack and Unstack
print("🌟 Advanced Reshaping - Stack & Unstack:")
print("💡 Stack/Unstack work with MultiIndex DataFrames")

# Create MultiIndex DataFrame from our pivot table
multi_pivot = df.pivot_table(
    values=["total_sales", "customer_rating"],
    index="product",
    columns="sales_rep",
    aggfunc="mean",
).round(2)

print("📊 MultiIndex pivot table:")
print(multi_pivot.head())
print()

# Stack - convert columns to rows (wide to long)
stacked = multi_pivot.stack()
print("📚 Stacked format (columns → rows):")
print(stacked.head(8))
print()

# Unstack - convert rows to columns (long to wide)
unstacked = stacked.unstack(level=0)  # level=0 means unstack the first index level
print("📖 Unstacked format (rows → columns):")
print(unstacked.head())
print()

"""
🎯 Data Reshaping Recap:
- Pivot tables create cross-tabulations for business reporting
- Melting converts wide format to long format (better for analysis/visualization)
- Wide format: each variable in separate columns
- Long format: variables stacked in rows
- Stack/Unstack work with MultiIndex for complex reshaping
- Choose format based on your analysis or visualization needs
"""

# ==============================================================================
# 📅 SECTION 8: Date & Time Operations - Temporal Analysis
# ==============================================================================

"""
🎯 Why DateTime Analysis Matters:
Time-based patterns reveal crucial business insights:
- Seasonal trends and cyclical patterns
- Growth rates and momentum
- Day-of-week and hour-of-day effects
- Time-to-event analysis
- Forecasting and planning

Master datetime operations to unlock temporal insights!
"""

print("📅 SECTION 8: DATE & TIME OPERATIONS - TEMPORAL ANALYSIS")
print("=" * 58)
print("🎯 Why DateTime? Discover time-based patterns and trends!")
print()

# 🗓️ DateTime Feature Engineering
print("🗓️ DateTime Feature Engineering:")
print("💡 Extract meaningful components from dates for analysis")

# Our dataset already has dates - let's extract useful features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["month_name"] = df["date"].dt.month_name()  # January, February, etc.
df["day"] = df["date"].dt.day
df["day_name"] = df["date"].dt.day_name()  # Monday, Tuesday, etc.
df["day_of_year"] = df["date"].dt.dayofyear  # 1-365
df["week_of_year"] = df["date"].dt.isocalendar().week
df["is_weekend"] = df["date"].dt.weekday >= 5  # Saturday=5, Sunday=6
df["is_month_start"] = df["date"].dt.is_month_start
df["quarter"] = df["date"].dt.quarter

print("📊 Extracted date features (first 5 rows):")
date_features = df[["date", "day_name", "is_weekend", "quarter", "total_sales"]].head()
print(date_features)
print()

# 📈 Time-Based Analysis
print("📈 Time-Based Business Analysis:")

# Daily sales trend
daily_sales = (
    df.groupby("date")
    .agg({"total_sales": "sum", "customer_rating": "mean", "quantity": "sum"})
    .round(2)
)

daily_sales.columns = ["daily_revenue", "daily_rating", "daily_units"]
print("📊 Daily sales performance:")
print(daily_sales.head(7))
print()

# Weekend vs Weekday analysis
weekend_analysis = (
    df.groupby("is_weekend")
    .agg(
        {
            "total_sales": ["mean", "sum", "count"],
            "customer_rating": "mean",
            "quantity": "mean",
        }
    )
    .round(2)
)

weekend_analysis.columns = [
    "avg_sale",
    "total_sales",
    "transaction_count",
    "avg_rating",
    "avg_quantity",
]

print("📊 Weekend vs Weekday Performance:")
for is_weekend, label in [(False, "Weekday"), (True, "Weekend")]:
    if is_weekend in weekend_analysis.index:
        metrics = weekend_analysis.loc[is_weekend]
        print(
            f"{label:>8}: Avg Sale=${metrics['avg_sale']:.2f}, "
            f"Total=${metrics['total_sales']:.2f}, "
            f"Rating={metrics['avg_rating']:.2f}"
        )
print()

# Day of week analysis
day_performance = (
    df.groupby("day_name")
    .agg({"total_sales": ["mean", "sum"], "customer_rating": "mean"})
    .round(2)
)

day_performance.columns = ["avg_daily_sale", "total_sales", "avg_rating"]

# Reorder by actual day sequence (Monday first)
day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
day_performance = day_performance.reindex(
    [day for day in day_order if day in day_performance.index]
)

print("📅 Performance by Day of Week:")
print(day_performance)
print()

# 🕐 Time Series Analysis with Rolling Windows
print("🕐 Rolling Window Analysis - Smooth Out Fluctuations:")
print("💡 Rolling windows help identify trends in noisy time series data")

# Calculate rolling averages to smooth daily fluctuations
daily_sales["sales_3day_avg"] = (
    daily_sales["daily_revenue"].rolling(window=3, center=True).mean().round(2)
)
daily_sales["sales_5day_avg"] = (
    daily_sales["daily_revenue"].rolling(window=5, center=True).mean().round(2)
)

# Calculate rolling statistics
daily_sales["sales_3day_max"] = (
    daily_sales["daily_revenue"].rolling(window=3, center=True).max()
)
daily_sales["sales_volatility"] = (
    daily_sales["daily_revenue"].rolling(window=3).std().round(2)
)

print("📈 Daily sales with rolling averages:")
rolling_columns = [
    "daily_revenue",
    "sales_3day_avg",
    "sales_5day_avg",
    "sales_volatility",
]
print(daily_sales[rolling_columns].head(8))
print()

# 📊 Time-based Business Insights
print("📊 Time-Based Business Insights:")

# Calculate growth rates
daily_sales["revenue_pct_change"] = daily_sales["daily_revenue"].pct_change() * 100
daily_sales["revenue_cumsum"] = daily_sales["daily_revenue"].cumsum()

print("📈 Revenue growth and cumulative totals:")
growth_data = daily_sales[
    ["daily_revenue", "revenue_pct_change", "revenue_cumsum"]
].head(8)
print(growth_data.round(2))
print()

# Time period comparisons
first_week = df[df["date"] <= "2024-01-07"]["total_sales"].sum()
second_week = df[df["date"] > "2024-01-07"]["total_sales"].sum()
week_growth = ((second_week - first_week) / first_week) * 100

print("📊 Week-over-Week Analysis:")
print(f"First week revenue: ${first_week:.2f}")
print(f"Second week revenue: ${second_week:.2f}")
print(f"Week-over-week growth: {week_growth:+.1f}%")
print()

# 🎯 Advanced DateTime Operations
print("🎯 Advanced DateTime Operations:")

# Date arithmetic
df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
df["days_until_month_end"] = (df["date"] + pd.offsets.MonthEnd(0) - df["date"]).dt.days

# Business day calculations (excluding weekends)
df["business_days_since_start"] = df["date"].map(
    lambda x: pd.bdate_range(start=df["date"].min(), end=x).shape[0] - 1
)

print("📅 Advanced date calculations (first 5 rows):")
advanced_dates = df[
    ["date", "days_since_start", "days_until_month_end", "business_days_since_start"]
].head()
print(advanced_dates)
print()

"""
🎯 DateTime Operations Recap:
- .dt accessor extracts date components (year, month, day, etc.)
- Boolean conditions identify patterns (weekends, month-end, etc.)
- Rolling windows smooth time series and identify trends
- pct_change() calculates period-over-period growth rates
- cumsum() creates running totals for cumulative analysis
- Date arithmetic enables time-based comparisons and calculations
"""

# ==============================================================================
# 🔗 SECTION 9: Merging DataFrames - Combining Data Sources
# ==============================================================================

"""
🎯 Why Data Merging Is Critical:
Real business data spans multiple systems:
- Customer data in CRM systems
- Transaction data in e-commerce platforms
- Product catalogs in inventory systems
- Financial data in accounting systems

Merging combines related data for comprehensive analysis!
"""

print("🔗 SECTION 9: MERGING DATAFRAMES - COMBINING DATA SOURCES")
print("=" * 58)
print("🎯 Why Merge? Combine related data from different sources!")
print()

# 🏭 Create Additional Data Sources
print("🏭 Creating Related Data Sources:")

# Product catalog data (simulates inventory management system)
product_catalog = pd.DataFrame(
    {
        "product": ["Laptop", "Mouse", "Keyboard", "Monitor", "Tablet"],
        "brand": ["TechCorp", "QuickClick", "TypeMaster", "ViewPro", "TouchTech"],
        "warranty_months": [24, 12, 18, 36, 12],
        "weight_kg": [2.5, 0.1, 0.8, 8.2, 0.6],
        "launch_year": [2022, 2023, 2021, 2020, 2024],
        "category_detailed": [
            "Gaming Laptop",
            "Wireless Mouse",
            "Mechanical Keyboard",
            "4K Monitor",
            "Android Tablet",
        ],
    }
)

print("🗃️ Product Catalog (Inventory System):")
print(product_catalog)
print()

# Customer data (simulates CRM system)
customer_data = pd.DataFrame(
    {
        "sales_rep": ["Alice", "Bob", "Charlie"],
        "region": ["North", "South", "East"],
        "years_experience": [5, 3, 7],
        "monthly_target": [15000, 12000, 18000],
        "commission_rate": [0.05, 0.04, 0.06],
    }
)

print("👥 Sales Rep Data (CRM System):")
print(customer_data)
print()

# 🔗 Different Types of Merges
print("🔗 Understanding Different Merge Types:")

# INNER JOIN - Only matching records from both datasets
print("🎯 Inner Join - Only products that exist in both datasets:")
inner_merged = df.merge(product_catalog, on="product", how="inner")
print(f"Original sales records: {len(df)}")
print(f"After inner join: {len(inner_merged)} (only matching products)")
print(inner_merged[["product", "brand", "total_sales"]].head(3))
print()

# LEFT JOIN - Keep all sales records, add product info where available
print("🎯 Left Join - Keep all sales, add product info where available:")
left_merged = df.merge(product_catalog, on="product", how="left")
print(f"After left join: {len(left_merged)} (same as original)")
print("Products without catalog info show NaN:")
print(left_merged[["product", "brand", "total_sales"]].head(3))
print()

# RIGHT JOIN - Keep all products, add sales where available
print("🎯 Right Join - Keep all products, add sales where available:")
right_merged = df.merge(product_catalog, on="product", how="right")
print(f"After right join: {len(right_merged)} (includes unsold products)")
products_without_sales = right_merged[right_merged["total_sales"].isna()]
if len(products_without_sales) > 0:
    print("Products without sales:")
    print(products_without_sales[["product", "brand", "total_sales"]])
print()

# OUTER JOIN - Keep all records from both datasets
print("🎯 Outer Join - Keep everything from both datasets:")
outer_merged = df.merge(product_catalog, on="product", how="outer")
print(f"After outer join: {len(outer_merged)} (union of both datasets)")
print()

# 🔧 Advanced Merging Techniques
print("🔧 Advanced Merging Techniques:")

# Multiple-column merging with suffix handling
enriched_sales = df.merge(
    product_catalog, on="product", how="left", suffixes=("_sales", "_catalog")
)

# Chain multiple merges
fully_enriched = enriched_sales.merge(customer_data, on="sales_rep", how="left")

print("🎯 Fully Enriched Dataset (first 3 records):")
enriched_columns = [
    "product",
    "brand",
    "sales_rep",
    "region",
    "total_sales",
    "warranty_months",
]
print(fully_enriched[enriched_columns].head(3))
print()

# 📊 Analysis with Enriched Data
print("📊 Business Analysis with Enriched Data:")

# Brand performance analysis
brand_performance = (
    fully_enriched.groupby("brand")
    .agg(
        {
            "total_sales": ["sum", "mean", "count"],
            "customer_rating": "mean",
            "warranty_months": "first",  # Same for all products of this brand
        }
    )
    .round(2)
)

brand_performance.columns = [
    "total_revenue",
    "avg_transaction",
    "sales_count",
    "avg_rating",
    "warranty_months",
]
brand_performance = brand_performance.sort_values("total_revenue", ascending=False)

print("🏢 Brand Performance Analysis:")
print(brand_performance)
print()

# Regional performance analysis
regional_performance = (
    fully_enriched.groupby("region")
    .agg(
        {
            "total_sales": ["sum", "mean"],
            "customer_rating": "mean",
            "sales_rep": "nunique",  # Count unique sales reps per region
        }
    )
    .round(2)
)

regional_performance.columns = [
    "total_revenue",
    "avg_transaction",
    "avg_rating",
    "sales_reps",
]

print("🗺️ Regional Performance Analysis:")
print(regional_performance)
print()

# Sales rep performance vs targets
rep_performance = (
    fully_enriched.groupby(["sales_rep", "monthly_target"])
    .agg({"total_sales": "sum"})
    .round(2)
)

rep_performance["target_achievement"] = (
    rep_performance["total_sales"]
    / rep_performance.index.get_level_values("monthly_target")
    * 100
).round(1)

print("🎯 Sales Rep Performance vs Monthly Targets:")
print(rep_performance)
print()

# 🔍 Data Quality Checks After Merging
print("🔍 Data Quality Checks After Merging:")

print(f"✅ Records before merging: {len(df)}")
print(f"✅ Records after enrichment: {len(fully_enriched)}")
print(f"✅ No duplicate records: {not fully_enriched.duplicated().any()}")

# Check for missing values in key columns
missing_after_merge = (
    fully_enriched[["brand", "region", "warranty_months"]].isnull().sum()
)
print("🔍 Missing values after merging:")
print(missing_after_merge)
print()

"""
🎯 Data Merging Recap:
- Inner join: only matching records from both datasets
- Left join: keep all from left dataset, add matching from right
- Right join: keep all from right dataset, add matching from left
- Outer join: keep all records from both datasets
- Chain multiple merges for comprehensive data enrichment
- Always check data quality and record counts after merging
- Use suffixes to handle overlapping column names
"""

# ==============================================================================
# 🎯 SECTION 10: Advanced Transformations & Performance Tips
# ==============================================================================

"""
🎯 Why Advanced Transformations Matter:
Complex business rules often require:
- Custom functions applied row-wise or column-wise
- Conditional logic based on multiple criteria
- Performance optimization for large datasets
- Efficient memory usage and vectorized operations

Master these techniques for production-ready data analysis!
"""

print("🎯 SECTION 10: ADVANCED TRANSFORMATIONS & PERFORMANCE")
print("=" * 55)
print("🎯 Why Advanced? Handle complex business logic efficiently!")
print()

# 🧠 Custom Functions with .apply()
print("🧠 Custom Business Logic with .apply():")
print("💡 .apply() runs custom functions on rows (axis=1) or columns (axis=0)")


def categorize_transaction(row):
    """
    Multi-criteria business logic for transaction categorization
    This demonstrates complex conditional logic
    """
    # High-value transactions get priority classification
    if row["total_sales"] > 1000:
        return "Premium"
    # High satisfaction customers are valuable even with lower sales
    elif row["customer_rating"] >= 4.5:
        return "High Satisfaction"
    # Bulk purchases indicate wholesale customers
    elif row["quantity"] >= 3:
        return "Bulk Purchase"
    # Weekend sales might have different characteristics
    elif row["is_weekend"]:
        return "Weekend Sale"
    # New products (launched recently) need special attention
    elif row.get("launch_year", 2020) >= 2023:
        return "New Product"
    else:
        return "Standard"


# Apply function to each row (axis=1)
fully_enriched["transaction_category"] = fully_enriched.apply(
    categorize_transaction, axis=1
)

print("🏷️ Transaction Categorization Results:")
category_counts = fully_enriched["transaction_category"].value_counts()
print(category_counts)
print()

# Show examples of each category
print("📋 Examples by Category:")
for category in category_counts.head(3).index:
    example = fully_enriched[fully_enriched["transaction_category"] == category].iloc[0]
    print(
        f"{category}: {example['product']} - ${example['total_sales']:.2f} (Rating: {example['customer_rating']})"
    )
print()

# 🚀 Vectorized Operations for Performance
print("🚀 Performance Optimization - Vectorized vs Loop Operations:")
print("💡 Vectorized operations are 10-100x faster than loops!")

# Create larger dataset for performance demonstration
np.random.seed(42)
large_df = pd.DataFrame(
    {"value": np.random.rand(10000), "multiplier": np.random.rand(10000)}
)


# Vectorized approach (FAST)
start_time = time.time()
large_df["result_vectorized"] = large_df["value"] * large_df["multiplier"] * 100
vectorized_time = time.time() - start_time

print(f"⚡ Vectorized operation on 10K records: {vectorized_time:.4f} seconds")

# Show the power of vectorized string operations
sample_text = pd.Series(
    ["Hello World", "DATA science", "python PANDAS", "Machine Learning"]
)
print("\n📝 Vectorized String Operations:")
print(f"Original: {sample_text.tolist()}")
print(f"Lowercase: {sample_text.str.lower().tolist()}")
print(f"Word count: {sample_text.str.split().str.len().tolist()}")
print(f"Contains 'data': {sample_text.str.contains('data', case=False).tolist()}")
print()

# 📊 Memory Optimization Techniques
print("📊 Memory Optimization for Large Datasets:")

# Check memory usage
memory_before = fully_enriched.memory_usage(deep=True).sum()
print(f"Memory usage before optimization: {memory_before:,} bytes")

# Optimize data types
memory_optimized = fully_enriched.copy()

# Convert appropriate columns to categorical (saves memory for repeated strings)
categorical_columns = [
    "product",
    "sales_rep",
    "brand",
    "region",
    "transaction_category",
]
for col in categorical_columns:
    if col in memory_optimized.columns:
        memory_optimized[col] = memory_optimized[col].astype("category")

# Convert appropriate numeric columns to smaller types
if "quantity" in memory_optimized.columns:
    memory_optimized["quantity"] = memory_optimized["quantity"].astype(
        "int8"
    )  # 1-5 range fits in int8

memory_after = memory_optimized.memory_usage(deep=True).sum()
memory_savings = ((memory_before - memory_after) / memory_before) * 100

print(f"Memory usage after optimization: {memory_after:,} bytes")
print(f"Memory savings: {memory_savings:.1f}%")
print()

# 🔄 Advanced Aggregation Patterns
print("🔄 Advanced Aggregation Patterns:")

# Named aggregations (pandas 0.25+)
advanced_agg = (
    fully_enriched.groupby("brand")
    .agg(
        total_revenue=("total_sales", "sum"),
        avg_rating=("customer_rating", "mean"),
        transaction_count=("total_sales", "count"),
        rating_std=("customer_rating", "std"),
        max_transaction=("total_sales", "max"),
    )
    .round(2)
)

print("📊 Advanced Aggregation with Named Functions:")
print(advanced_agg)
print()


# Conditional aggregation
def conditional_stats(group):
    """Custom aggregation function with conditional logic"""
    return pd.Series(
        {
            "high_value_count": (group["total_sales"] > 500).sum(),
            "high_value_revenue": group[group["total_sales"] > 500][
                "total_sales"
            ].sum(),
            "avg_high_value": group[group["total_sales"] > 500]["total_sales"].mean()
            if (group["total_sales"] > 500).any()
            else 0,
        }
    )


conditional_results = fully_enriched.groupby("brand").apply(conditional_stats).round(2)
print("🎯 Conditional Aggregation - High-Value Transactions by Brand:")
print(conditional_results)
print()

# 📈 Performance Tips Summary
print("📈 Performance Best Practices:")
performance_tips = [
    "✅ Use vectorized operations instead of loops",
    "✅ Convert repeated strings to categorical data type",
    "✅ Use appropriate numeric data types (int8, int16 vs int64)",
    "✅ Filter data early to reduce memory usage",
    "✅ Use .loc for label-based selection, .iloc for position-based",
    "✅ Avoid chained assignment (use .copy() when modifying)",
    "✅ Use .query() for complex boolean filtering",
    "✅ Consider chunking for very large datasets",
]

for tip in performance_tips:
    print(tip)
print()

# 🎯 Common Pandas Pitfalls and Solutions
print("🎯 Common Pandas Pitfalls to Avoid:")

print("❌ Chained Assignment (can cause warnings):")
print("   df['col1'][df['col2'] > 5] = 'new_value'  # BAD")
print("✅ Use .loc instead:")
print("   df.loc[df['col2'] > 5, 'col1'] = 'new_value'  # GOOD")
print()

print("❌ Inefficient string operations:")
print("   df['col'].apply(lambda x: x.lower())  # SLOWER")
print("✅ Use vectorized string methods:")
print("   df['col'].str.lower()  # FASTER")
print()

print("❌ Using loops for element-wise operations:")
print(
    "   for i in range(len(df)): df.loc[i, 'result'] = df.loc[i, 'a'] * 2  # VERY SLOW"
)
print("✅ Use vectorized operations:")
print("   df['result'] = df['a'] * 2  # VERY FAST")
print()

"""
🎯 Advanced Transformations Recap:
- .apply() enables custom business logic on rows or columns
- Vectorized operations are dramatically faster than loops
- Memory optimization through data types and categorical encoding
- Named aggregations improve code readability
- Avoid chained assignment, use .loc for clarity
- Vectorized string operations (.str) beat apply() for text processing
- Always profile performance on realistic data sizes
"""

# ==============================================================================
# 🎯 FINAL SECTION: Key Takeaways & Next Steps
# ==============================================================================

print("🎯 COMPREHENSIVE BUSINESS INSIGHTS FROM OUR ANALYSIS")
print("=" * 55)

# Extract comprehensive business insights
insights = {
    "total_revenue": fully_enriched["total_sales"].sum(),
    "total_transactions": len(fully_enriched),
    "avg_transaction": fully_enriched["total_sales"].mean(),
    "top_product": fully_enriched.groupby("product")["total_sales"].sum().idxmax(),
    "top_brand": fully_enriched.groupby("brand")["total_sales"].sum().idxmax(),
    "top_sales_rep": fully_enriched.groupby("sales_rep")["total_sales"].sum().idxmax(),
    "top_region": fully_enriched.groupby("region")["total_sales"].sum().idxmax(),
    "avg_rating": fully_enriched["customer_rating"].mean(),
    "weekend_premium": (
        fully_enriched[fully_enriched["is_weekend"]]["total_sales"].mean()
        / fully_enriched[~fully_enriched["is_weekend"]]["total_sales"].mean()
        - 1
    )
    * 100,
    "high_value_transactions": (fully_enriched["total_sales"] > 500).sum(),
    "premium_category_pct": (fully_enriched["transaction_category"] == "Premium").mean()
    * 100,
}

print("🏆 EXECUTIVE DASHBOARD:")
print(f"   💰 Total Revenue: ${insights['total_revenue']:,.2f}")
print(f"   📊 Total Transactions: {insights['total_transactions']:,}")
print(f"   💳 Average Transaction: ${insights['avg_transaction']:.2f}")
print(f"   🥇 Best Product: {insights['top_product']}")
print(f"   🏢 Leading Brand: {insights['top_brand']}")
print(f"   👨‍💼 Top Sales Rep: {insights['top_sales_rep']}")
print(f"   🗺️ Best Region: {insights['top_region']}")
print(f"   ⭐ Average Rating: {insights['avg_rating']:.2f}/5.0")
print(f"   📅 Weekend Sales Premium: {insights['weekend_premium']:+.1f}%")
print(f"   💎 High-Value Transactions: {insights['high_value_transactions']}")
print(f"   🏆 Premium Category: {insights['premium_category_pct']:.1f}%")
print()

# 💾 Export Options for Real-World Usage
print("💾 EXPORT YOUR ANALYSIS:")
print("🔧 Ready-to-use export commands:")
print()
print("📊 CSV Export:")
print("   fully_enriched.to_csv('complete_sales_analysis.csv', index=False)")
print()
print("📈 Excel Export (Multiple Sheets):")
print("   with pd.ExcelWriter('sales_dashboard.xlsx', engine='openpyxl') as writer:")
print("       fully_enriched.to_excel(writer, sheet_name='Raw_Data', index=False)")
print("       brand_performance.to_excel(writer, sheet_name='Brand_Analysis')")
print("       daily_sales.to_excel(writer, sheet_name='Daily_Trends')")
print()
print("🌐 JSON Export:")
print(
    "   fully_enriched.to_json('sales_data.json', orient='records', date_format='iso')"
)
print()
print("🔍 HTML Report:")
print(
    "   fully_enriched.to_html('sales_report.html', index=False, table_id='sales_table')"
)
print()

"""
==============================================================================
🎯 COMPREHENSIVE KEY TAKEAWAYS
==============================================================================


📊 Pandas DataFrames: Your main tool for structured data
🔍 Indexing: .loc/.iloc for flexible selection
🧹 Data Cleaning: Handle missing & messy data early
📊 GroupBy & Aggregation: Summarize insights fast
🔗 Joins & Merges: Combine datasets with merge()
🔄 Reshaping: Use melt(), pivot(), stack/unstack()
🧮 NumPy Arrays: Fast, efficient numerical ops
🚀 Broadcasting & Vectorization: Speed up calculations
⚡ Optimize: Use vectorized

"""

"""
===============================================================================
🎯 FINAL RECAP - ESSENTIAL DATA MANIPULATION CONCEPTS
===============================================================================

This tutorial covered the complete data manipulation workflow:

1. 📊 NumPy Foundations
   - Fast numerical arrays and vectorized operations
   - Boolean indexing and mathematical functions

2. 🏗️ DataFrame Mastery
   - Creating, inspecting, and understanding DataFrame structure
   - Mixed data types and memory optimization

3. 🔍 Data Selection Excellence
   - .loc vs .iloc: when to use label-based vs position-based selection
   - Boolean filtering with complex conditions
   - Efficient data subsetting techniques

4. 🧹 Professional Data Cleaning
   - Handling missing values systematically
   - Text standardization and data type conversion
   - Data validation and quality checks

5. 📝 String Operations Power
   - Vectorized text processing with .str accessor
   - Pattern matching with regular expressions
   - Text analysis and feature extraction

6. 📊 Business Intelligence with GroupBy
   - Split-apply-combine methodology
   - Multiple aggregation functions and custom metrics
   - Multi-level grouping for hierarchical analysis

7. 🔄 Data Reshaping Mastery
   - Pivot tables for cross-tabulation
   - Melting for format conversion (wide ↔ long)
   - When and why to reshape data

8. 📅 Temporal Analysis Skills
   - Date component extraction and feature engineering
   - Rolling windows and trend analysis
   - Time-based business calculations

9. 🔗 Data Integration Techniques
   - Four types of joins and when to use each
   - Combining data from multiple sources
   - Data quality verification after merging

10. 🚀 Performance Optimization
    - Vectorization vs loops performance comparison
    - Memory optimization strategies
    - Common pitfalls and best practices

The key to mastering data manipulation is consistent practice with real datasets.
Start small, build complexity gradually, and always focus on solving actual
business problems. These foundational skills will serve you throughout your
entire data science career!

Happy analyzing! 🎯
===============================================================================
"""
