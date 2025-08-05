"""
📊 Data Manipulation Reference & Cheatsheet
NumPy, Pandas, and real-world data cleaning examples
"""

import numpy as np
import pandas as pd
from datetime import datetime  # , timedelta

# import json
from io import StringIO

# =============================================================================
# 🔢 NUMPY FUNDAMENTALS
# =============================================================================

print("🔢 NumPy Array Operations")
print("=" * 50)

# -- Array Creation & Basic Operations --
# Different ways to create arrays
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
range_arr = np.arange(0, 20, 2)  # start, stop, step
linspace_arr = np.linspace(0, 10, 5)  # start, stop, num_points

print(f"📏 1D Array: {arr1d}")
print(f"📐 2D Array:\n{arr2d}")
print(f"🔍 Array shape: {arr2d.shape}, dtype: {arr2d.dtype}")
print(f"📊 Range array: {range_arr}")
print(f"📈 Linspace: {linspace_arr}")

# -- Indexing & Slicing --
matrix = np.random.randint(1, 10, (4, 5))
print(f"\n🎲 Random Matrix:\n{matrix}")

# Various indexing techniques
print(f"🔍 Element at [1,2]: {matrix[1, 2]}")
print(f"🔍 First row: {matrix[0, :]}")
print(f"🔍 Last column: {matrix[:, -1]}")
print(f"🔍 Submatrix [1:3, 2:4]:\n{matrix[1:3, 2:4]}")

# Boolean indexing (very powerful)
mask = matrix > 5
print(f"🔍 Elements > 5: {matrix[mask]}")
print(f"🔍 Count > 5: {np.sum(mask)}")

# -- Array Reshaping & Manipulation --
original = np.arange(12)
reshaped = original.reshape(3, 4)
flattened = reshaped.flatten()
transposed = reshaped.T

print(f"\n🔄 Original: {original}")
print(f"🔄 Reshaped (3x4):\n{reshaped}")
print(f"🔄 Transposed:\n{transposed}")
print(f"🔄 Flattened: {flattened}")

# -- Mathematical Operations --
arr_a = np.array([1, 2, 3, 4])
arr_b = np.array([10, 20, 30, 40])

print(f"\n🧮 Array A: {arr_a}")
print(f"🧮 Array B: {arr_b}")
print(f"➕ Addition: {arr_a + arr_b}")
print(f"✖️ Element-wise multiply: {arr_a * arr_b}")
print(f"🔢 Dot product: {np.dot(arr_a, arr_b)}")
print(f"📊 Statistics A: mean={np.mean(arr_a):.2f}, std={np.std(arr_a):.2f}")

# -- Broadcasting Example --
matrix_3x3 = np.random.randint(1, 10, (3, 3))
row_vector = np.array([1, 2, 3])

print("\n📡 Broadcasting Example:")
print(f"Matrix:\n{matrix_3x3}")
print(f"Row vector: {row_vector}")
print(f"Matrix + Row (broadcasting):\n{matrix_3x3 + row_vector}")

# =============================================================================
# 🐼 PANDAS PROFICIENCY
# =============================================================================

print("\n\n🐼 Pandas Data Manipulation")
print("=" * 50)

# -- DataFrame Creation --
# From dictionary
sales_data = {
    "date": pd.date_range("2024-01-01", periods=10, freq="D"),
    "product": ["Laptop", "Mouse", "Keyboard", "Monitor", "Tablet"] * 2,
    "category": [
        "Electronics",
        "Accessories",
        "Accessories",
        "Electronics",
        "Electronics",
    ]
    * 2,
    "price": [999, 25, 75, 300, 450, 1200, 30, 80, 280, 500],
    "quantity": np.random.randint(1, 10, 10),
    "customer_rating": np.random.uniform(3.0, 5.0, 10),
}

df = pd.DataFrame(sales_data)
df["total_sales"] = df["price"] * df["quantity"]
print(f"📊 Sales DataFrame:\n{df.head()}")
print(f"📏 Shape: {df.shape}, Memory usage: {df.memory_usage().sum()} bytes")

# -- Basic DataFrame Operations --
print("\n🔍 DataFrame Info:")
print(f"📊 Data types:\n{df.dtypes}")
print(f"📈 Numerical summary:\n{df.describe()}")
print(f"🏷️ Categories: {df['category'].value_counts().to_dict()}")

# -- Filtering & Selection --
# Multiple filtering techniques
high_value = df[df["total_sales"] > 500]
electronics = df[df["category"] == "Electronics"]
recent_high_rated = df[(df["date"] >= "2024-01-05") & (df["customer_rating"] >= 4.0)]

print("\n🔍 Filtering Results:")
print(
    f"💰 High value sales ({len(high_value)} rows): avg = ${high_value['total_sales'].mean():.2f}"
)
print(f"📱 Electronics ({len(electronics)} rows): {electronics['product'].unique()}")
print(f"⭐ Recent high-rated ({len(recent_high_rated)} rows)")

# -- GroupBy Operations --
category_stats = (
    df.groupby("category")
    .agg(
        {
            "total_sales": ["sum", "mean", "count"],
            "customer_rating": "mean",
            "quantity": "sum",
        }
    )
    .round(2)
)

print(f"\n📊 Category Statistics:\n{category_stats}")

# Product performance analysis
product_perf = (
    df.groupby("product")
    .agg({"total_sales": "sum", "quantity": "sum", "customer_rating": "mean"})
    .sort_values("total_sales", ascending=False)
)

print(f"\n🏆 Product Performance:\n{product_perf}")

# -- Pivot Tables --
pivot_table = df.pivot_table(
    values="total_sales",
    index="product",
    columns="category",
    aggfunc="sum",
    fill_value=0,
)
print(f"\n🔄 Pivot Table:\n{pivot_table}")

# -- File I/O Examples --
# Create sample CSV data in memory
csv_data = """name,age,department,salary,join_date
Alice Johnson,28,Engineering,85000,2022-03-15
Bob Smith,35,Marketing,65000,2021-07-22
Charlie Brown,42,Engineering,95000,2020-01-10
Diana Prince,31,Sales,72000,2022-11-05
Eve Wilson,29,Marketing,68000,2023-02-14"""

# Read from string (simulating CSV file)
employees_df = pd.read_csv(StringIO(csv_data))
employees_df["join_date"] = pd.to_datetime(employees_df["join_date"])
employees_df["years_employed"] = (
    datetime.now() - employees_df["join_date"]
).dt.days / 365.25

print(f"\n👥 Employee Data:\n{employees_df}")

# JSON handling
json_data = [
    {"id": 1, "name": "Product A", "metrics": {"views": 1000, "sales": 50}},
    {"id": 2, "name": "Product B", "metrics": {"views": 1500, "sales": 75}},
    {"id": 3, "name": "Product C", "metrics": {"views": 800, "sales": 30}},
]

# Normalize nested JSON
json_df = pd.json_normalize(json_data)
print(f"\n📄 JSON Normalized:\n{json_df}")

# -- Advanced DataFrame Operations --
# Rolling windows for time series
df_sorted = df.sort_values("date")
df_sorted["rolling_avg_sales"] = df_sorted["total_sales"].rolling(window=3).mean()
df_sorted["cumulative_sales"] = df_sorted["total_sales"].cumsum()

print(
    f"\n📈 Time Series Features:\n{df_sorted[['date', 'total_sales', 'rolling_avg_sales', 'cumulative_sales']].head(7)}"
)

# =============================================================================
# 🧹 DATA CLEANING & REAL-WORLD MANIPULATIONS
# =============================================================================

print("\n\n🧹 Data Cleaning Operations")
print("=" * 50)

# -- Create Messy Dataset --
messy_data = {
    "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "name": [
        "John Doe",
        "jane smith",
        "BOB JOHNSON",
        None,
        "Alice Brown",
        "charlie wilson",
        "",
        "Eve Davis",
        "Frank Miller",
        "Grace Lee",
    ],
    "email": [
        "john@email.com",
        "JANE@EMAIL.COM",
        "bob@email",
        "alice@email.com",
        "alice@email.com",
        "charlie@email.com",
        None,
        "eve@email.com",
        "frank@email.com",
        "grace@email.com",
    ],
    "age": [25, 30, None, 35, 28, 45, 22, None, 38, 29],
    "salary": [
        "50000",
        "60,000",
        "75000",
        None,
        "55000",
        "80,000",
        "45000",
        "70000",
        None,
        "52000",
    ],
    "phone": [
        "123-456-7890",
        "(234) 567-8901",
        "345.678.9012",
        "456-789-0123",
        "567-890-1234",
        None,
        "678-901-2345",
        "789-012-3456",
        "890-123-4567",
        "901-234-5678",
    ],
}

messy_df = pd.DataFrame(messy_data)
print(f"🗑️ Original Messy Data:\n{messy_df}")
print(f"❌ Missing values per column:\n{messy_df.isnull().sum()}")

# -- Data Cleaning Pipeline --
cleaned_df = messy_df.copy()

# 1. Handle missing and empty names
cleaned_df["name"] = cleaned_df["name"].replace("", None)  # Empty string to None
cleaned_df["name"] = cleaned_df["name"].fillna("Unknown Customer")

# 2. Standardize name formatting
cleaned_df["name"] = cleaned_df["name"].str.title()

# 3. Clean and validate emails
cleaned_df["email"] = cleaned_df["email"].str.lower().str.strip()
# Mark invalid emails (basic validation)
email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
cleaned_df["valid_email"] = cleaned_df["email"].str.match(email_pattern, na=False)

# 4. Handle salary column (remove commas, convert to numeric)
cleaned_df["salary"] = cleaned_df["salary"].str.replace(",", "").str.replace("$", "")
cleaned_df["salary"] = pd.to_numeric(cleaned_df["salary"], errors="coerce")

# 5. Fill missing ages with median
median_age = cleaned_df["age"].median()
cleaned_df["age"] = cleaned_df["age"].fillna(median_age)


# 6. Standardize phone numbers
def clean_phone(phone):
    if pd.isna(phone):
        return None
    # Remove all non-digits
    digits_only = "".join(filter(str.isdigit, str(phone)))
    if len(digits_only) == 10:
        return f"{digits_only[:3]}-{digits_only[3:6]}-{digits_only[6:]}"
    return phone  # Return original if can't standardize


cleaned_df["phone"] = cleaned_df["phone"].apply(clean_phone)

print(f"\n✅ Cleaned Data:\n{cleaned_df}")
print(f"✅ Missing values after cleaning:\n{cleaned_df.isnull().sum()}")

# -- Duplicate Detection & Handling --
# Add some duplicate records for demonstration
duplicate_rows = cleaned_df.iloc[[1, 4]].copy()  # Duplicate Jane and Alice
duplicate_rows["customer_id"] = [11, 12]  # Change IDs
extended_df = pd.concat([cleaned_df, duplicate_rows], ignore_index=True)

print("\n🔍 Duplicate Detection:")
print(f"📊 Total rows: {len(extended_df)}")

# Find duplicates based on email
email_duplicates = extended_df[
    extended_df.duplicated(["email"], keep=False)
].sort_values("email")
print(f"📧 Email duplicates:\n{email_duplicates[['customer_id', 'name', 'email']]}")

# Remove duplicates (keep first occurrence)
deduplicated_df = extended_df.drop_duplicates(["email"], keep="first")
print(f"🧹 After deduplication: {len(deduplicated_df)} rows")

# -- String Operations & Text Cleaning --
text_data = pd.DataFrame(
    {
        "comments": [
            "Great product! Highly recommended.",
            "Not bad, could be better... 3/5 stars",
            "AMAZING!!! Will buy again!!!",
            "Poor quality. Waste of money.",
            "Good value for money. Fast shipping.",
            None,
            "Excellent customer service. 5 stars!",
            "Average product, nothing special",
        ]
    }
)

# Text cleaning operations
text_data["comment_length"] = text_data["comments"].str.len()
text_data["word_count"] = text_data["comments"].str.split().str.len()
text_data["has_exclamation"] = text_data["comments"].str.contains("!", na=False)
text_data["sentiment_words"] = (
    text_data["comments"]
    .str.lower()
    .str.extract(r"(great|amazing|excellent|good|poor|bad|waste)", expand=False)
)

print(f"\n📝 Text Analysis:\n{text_data}")

# -- Advanced Data Transformations --
# Create categorical data
sales_categories = pd.cut(
    cleaned_df["salary"],
    bins=[0, 50000, 70000, float("inf")],
    labels=["Low", "Medium", "High"],
)
cleaned_df["salary_category"] = sales_categories

# Date operations
cleaned_df["created_date"] = pd.date_range(
    "2024-01-01", periods=len(cleaned_df), freq="D"
)
cleaned_df["day_of_week"] = cleaned_df["created_date"].dt.day_name()
cleaned_df["is_weekend"] = cleaned_df["created_date"].dt.weekday >= 5

print(
    f"\n📅 Enhanced Data with Categories:\n{cleaned_df[['name', 'salary', 'salary_category', 'day_of_week', 'is_weekend']].head()}"
)

# -- Data Export Examples --
print("\n💾 Data Export Examples:")
# In a real scenario, you'd export to actual files:
# cleaned_df.to_csv('cleaned_customers.csv', index=False)
# cleaned_df.to_excel('cleaned_customers.xlsx', index=False)
# cleaned_df.to_json('cleaned_customers.json', orient='records')

# For demo, show the export commands
export_summary = {
    "total_records": len(cleaned_df),
    "clean_emails": cleaned_df["valid_email"].sum(),
    "salary_categories": cleaned_df["salary_category"].value_counts().to_dict(),
    "avg_salary": cleaned_df["salary"].mean(),
}
print(f"📊 Export Summary: {export_summary}")

print("\n🎉 Data Manipulation Complete! Ready for visualization & ML.")
