"""
🚀 Python Fundamentals Tutorial
===================================

Welcome to your first step into Python programming! This beginner-friendly tutorial covers essential Python concepts,
focusing clearly on the "why" behind the syntax and usage, to help you start coding confidently.

🎯 What you'll learn:
- Python's dynamic typing system and core data types
- String manipulation for real-world data cleaning
- Control flow: making decisions and automating repetition
- Functions: organizing code into reusable, testable blocks
- Data structures: lists, dicts, sets, and tuples for organizing information
- Advanced collections from Python's standard library
- Error handling to make robust, crash-proof programs
- Essential algorithms for efficient problem-solving

🔧 How to use:
Everything here is self-contained, interactive, and executable.
Each section clearly explains concepts and shows practical code examples.
Run this file from top to bottom to see outputs and learn step-by-step.

📦 Requirements:
You’ll only need a handful of standard Python packages to run this and all other tutorials in the minicourse:
NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, and SciPy.

If you're setting this up from scratch, just run:

    pip install -r requirements.txt

🤖 Note:
This is part of the [Cheatsheet DS Minicourse] — a side project made with
some coding, some love, and some help from Claude AI.

Let's dive in! 🎯
"""


# ============================
# 📦 Imports and Setup
# ============================

# Python comes with tons of built-in modules that extend what you can do
# We import them at the top so they're available throughout our program
import random
from collections import Counter, defaultdict, namedtuple
from typing import List, Dict, Optional, Any

# Set random seed so our "random" examples are reproducible
random.seed(42)

print("🚀 Welcome to Python Fundamentals!")
print("=" * 40)

# ============================
# 📌 Variables and Data Types
# ============================

# Variables are like labeled containers that hold information
# Python figures out what type of data you're storing automatically (dynamic typing)
# This makes Python beginner-friendly compared to languages like Java or C++

print("\n📌 VARIABLES AND DATA TYPES")
print("-" * 28)

# Python's core data types - the building blocks of every program
name = "Alice"  # str (string) - text data
age = 28  # int (integer) - whole numbers
salary = 75000.50  # float - decimal numbers
is_employed = True  # bool (boolean) - True or False values

# f-strings are Python's modern, clean way to format text
# Put an 'f' before quotes and use {variable_name} to insert values
print(f"👋 Meet {name}:")
print(f"   Age: {age} years")
print(f"   Salary: ${salary:,.2f}")  # :,.2f adds commas and 2 decimal places
print(f"   Employed: {is_employed}")

# The type() function tells us what kind of data we're working with
# This is super useful for debugging and understanding your data
print("\n🔍 Data types:")
print(f"   '{name}' is a {type(name).__name__}")
print(f"   {age} is an {type(age).__name__}")
print(f"   {salary} is a {type(salary).__name__}")
print(f"   {is_employed} is a {type(is_employed).__name__}")

# Key insight: Python's dynamic typing means you don't declare types upfront
# The interpreter figures it out based on the value you assign

# ============================
# 🧵 Strings and Text Processing
# ============================

# Strings are crucial in data science because real-world data is often messy text:
# inconsistent capitalization, extra spaces, different formats, etc.
# String methods help us clean and standardize this data

print("\n🧵 STRINGS AND TEXT PROCESSING")
print("-" * 32)

# Realistic messy data that needs cleaning
first_name = "john"
last_name = "DOE"
company = "  Data Corp  "

# String methods for cleaning and formatting
full_name = (
    first_name.title() + " " + last_name.title()
)  # .title() capitalizes properly
email = f"{first_name.lower()}.{last_name.lower()}@datacorp.com"
clean_company = company.strip()  # .strip() removes leading/trailing whitespace

print("📝 String cleaning in action:")
print(f"   Raw data: '{first_name}', '{last_name}', '{company}'")
print(f"   Clean name: {full_name}")
print(f"   Generated email: {email}")
print(f"   Clean company: '{clean_company}'")

# Essential string methods you'll use constantly in data science
sample_text = "  Machine Learning is AWESOME!  "
print("\n🧹 Common text cleaning operations:")
print(f"   Original: '{sample_text}'")
print(f"   Stripped: '{sample_text.strip()}'")
print(f"   Lowercase: '{sample_text.strip().lower()}'")
print(f"   Split into words: {sample_text.strip().split()}")
print(f"   Character count: {len(sample_text.strip())}")

# Why this matters: Most data cleaning involves string manipulation
# Names, addresses, categories - they all need standardization

# ============================
# 🔀 Conditional Logic (if/elif/else)
# ============================

# Programs need to make decisions based on data conditions
# Conditionals let us execute different code paths based on what we find
# Essential for data validation, filtering, and categorizing information

print("\n🔀 CONDITIONAL LOGIC")
print("-" * 20)


def categorize_score(score):
    """
    🎯 Categorize a test score into letter grades

    This demonstrates:
    - if/elif/else structure for multiple conditions
    - Comparison operators (>=, <, etc.)
    - How to return different values based on logic
    """
    if score >= 90:
        return f"A ({score}%) - Excellent!"
    elif score >= 80:
        return f"B ({score}%) - Good work!"
    elif score >= 70:
        return f"C ({score}%) - Passing"
    elif score >= 60:
        return f"D ({score}%) - Needs improvement"
    else:
        return f"F ({score}%) - Failing"


# Test our conditional logic with various scores
test_scores = [95, 87, 72, 65, 45]
print("📊 Score categorization:")
for score in test_scores:
    result = categorize_score(score)
    print(f"   {result}")

# Conditionals are the foundation of data filtering and business logic
# You'll use them constantly to handle different data scenarios

# ============================
# 🔄 Loops and Iteration
# ============================

# Instead of copying and pasting code, loops let us repeat actions efficiently
# In data science, we use loops to process datasets, apply transformations,
# and generate reports across multiple items or time periods

print("\n🔄 LOOPS AND ITERATION")
print("-" * 22)

# Sample data: product inventory (simulating real business data)
products = ["laptop", "mouse", "keyboard", "monitor", "webcam"]
prices = [999, 25, 75, 300, 120]
stock_levels = [15, 50, 30, 8, 25]

print("📦 Inventory Report:")
print("   ID | Product     | Price | Stock | Status")
print("   ---|-------------|-------|-------|--------")

# enumerate() gives us both position (index) and item from a list
# zip() combines multiple lists element-by-element - super useful!
for i, (product, price, stock) in enumerate(zip(products, prices, stock_levels), 1):
    # Business logic: determine stock status based on inventory levels
    if stock < 10:
        status = "LOW"
    elif stock < 20:
        status = "OK"
    else:
        status = "GOOD"

    # Format output in a clean table (:{width}s formats strings with fixed width)
    print(f"   {i:2d} | {product:11s} | ${price:4d} | {stock:5d} | {status}")

# List comprehensions: Python's superpower for transforming data
# They're like loops but more concise and often faster
print("\n🚀 List comprehensions (Python's superpower):")

# Filter expensive products using comprehension
expensive_products = [
    product for product, price in zip(products, prices) if price > 100
]
print(f"   Expensive items: {expensive_products}")

# Calculate inventory values (price × stock) for each product
total_values = [price * stock for price, stock in zip(prices, stock_levels)]
print(f"   Inventory values: {total_values}")

# Find products that need restocking
low_stock_items = [
    product for product, stock in zip(products, stock_levels) if stock < 15
]
print(f"   Need restocking: {low_stock_items}")

# Loops are everywhere in data processing - use them to automate repetitive tasks

# ============================
# 🧠 Functions
# ============================

# Functions are like mini-programs that take input, do work, and return output
# They help us organize code, avoid repetition, and make our programs testable
# Think of them as reusable tools in your data science toolbox

print("\n🧠 FUNCTIONS")
print("-" * 12)


def analyze_sales(daily_sales: List[float], target: float = 1000.0) -> Dict[str, Any]:
    """
    📊 Analyze daily sales performance and return insights

    Parameters:
        daily_sales: List of daily sales amounts
        target: Target sales amount (defaults to 1000.0)

    Returns:
        Dictionary containing analysis results

    💡 Type hints (List[float], Dict[str, Any]) document what the function expects
    This makes code more readable and helps catch bugs early
    """
    total_sales = sum(daily_sales)
    avg_sales = total_sales / len(daily_sales)
    best_day = max(daily_sales)
    worst_day = min(daily_sales)
    days_above_target = sum(1 for sales in daily_sales if sales > target)

    # Return a dictionary with all our calculated insights
    return {
        "total": total_sales,
        "average": avg_sales,
        "best_day": best_day,
        "worst_day": worst_day,
        "target": target,
        "days_above_target": days_above_target,
        "success_rate": (days_above_target / len(daily_sales)) * 100,
    }


# Test our function with realistic sales data
weekly_sales = [850, 1200, 950, 1100, 800, 1300, 900]
analysis = analyze_sales(weekly_sales, target=1000)

print("📈 Sales Analysis Results:")
for key, value in analysis.items():
    if isinstance(value, float):
        print(f"   {key}: {value:.2f}")
    else:
        print(f"   {key}: {value}")

# Functions make your code modular and reusable across different datasets
# They're essential for organizing complex data analysis workflows

# ============================
# 🛡️ Error Handling
# ============================

# Real-world data is messy. Users make mistakes. Networks fail.
# Error handling (try/except) prevents programs from crashing and lets us
# respond gracefully to problems instead of failing completely

print("\n🛡️ ERROR HANDLING")
print("-" * 17)


def safe_calculate(operation: str, a: float, b: float) -> Optional[float]:
    """
    🔒 Perform calculations safely without crashing the program

    Returns the result if successful, None if something goes wrong
    Optional[float] means "either a float or None"
    """
    try:
        if operation == "divide":
            if b == 0:
                print("   ⚠️  Cannot divide by zero!")
                return None
            result = a / b
        elif operation == "square_root":
            if a < 0:
                print("   ⚠️  Cannot take square root of negative number!")
                return None
            result = a**0.5
        else:
            print(f"   ⚠️  Unknown operation: {operation}")
            return None

        print(f"   ✅ {operation}({a}, {b}) = {result:.2f}")
        return result

    except TypeError:
        print(f"   ❌ Invalid input types: {type(a)}, {type(b)}")
        return None
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return None


# Test our error handling with various scenarios
print("🧪 Testing error handling:")
safe_calculate("divide", 10, 2)  # Should work fine
safe_calculate("divide", 10, 0)  # Division by zero
safe_calculate("square_root", -4, 0)  # Negative square root
safe_calculate("unknown", 5, 3)  # Unknown operation

# Error handling makes your programs robust and user-friendly
# Always anticipate what could go wrong with your data

# ============================
# 📚 Data Structures
# ============================

# Data structures are containers that organize and store information
# Choosing the right structure makes your code faster and more readable
# Each structure has strengths for different types of problems

print("\n📚 DATA STRUCTURES")
print("-" * 18)

# === Lists: Ordered, changeable sequences ===
print("📋 Lists - Ordered and changeable:")
numbers = list(range(1, 11))  # Creates [1, 2, 3, ..., 10]
print(f"   Original: {numbers}")

# Common list operations you'll use frequently
numbers.append(11)  # Add to end
numbers.insert(0, 0)  # Insert at specific position
numbers.remove(5)  # Remove first occurrence of value
print(f"   Modified: {numbers}")

# List comprehensions for data transformation (very Pythonic!)
evens = [n for n in numbers if n % 2 == 0]
squares = [n**2 for n in numbers[:5]]  # First 5 elements squared
print(f"   Even numbers: {evens}")
print(f"   First 5 squared: {squares}")

# === Dictionaries: Key-value pairs ===
print("\n📖 Dictionaries - Key-value mappings:")
# Like a real dictionary where you look up definitions by word
# Perfect for structured data where you need fast lookups
sales_data = {"January": 15000, "February": 18000, "March": 12000, "April": 22000}
print(f"   Monthly sales: {sales_data}")

# Dictionary operations for analysis
sales_data["May"] = 25000  # Add new key-value pair
total_sales = sum(sales_data.values())
best_month = max(sales_data, key=sales_data.get)  # Find key with max value
print(f"   Total sales: ${total_sales:,}")
print(f"   Best month: {best_month} (${sales_data[best_month]:,})")

# Dictionary comprehensions for filtering data
high_performing = {
    month: amount for month, amount in sales_data.items() if amount > 15000
}
print(f"   High performers: {high_performing}")

# === Sets: Unique items only ===
print("\n🎯 Sets - Collections of unique items:")
# Perfect for removing duplicates or finding intersections/differences
team_skills = {"python", "sql", "excel", "tableau", "python", "sql"}  # duplicates!
print(f"   Team skills (duplicates removed): {team_skills}")

required_skills = {"python", "sql", "statistics"}
nice_to_have = {"tableau", "r", "machine_learning"}

# Set operations for comparing groups
print(f"   Required skills: {required_skills}")
print(f"   Skills we have: {team_skills & required_skills}")  # Intersection
print(f"   Skills we're missing: {required_skills - team_skills}")  # Difference
print(
    f"   All possible skills: {team_skills | required_skills | nice_to_have}"
)  # Union

# === Tuples: Immutable sequences ===
print("\n📌 Tuples - Unchangeable sequences:")
# Good for coordinates, database records, or data that shouldn't change
employee_record = ("Alice Johnson", "Data Analyst", 75000, "2022-01-15")
name, position, salary, start_date = employee_record  # Unpacking
print(f"   Employee: {name}, {position}, ${salary:,}, started {start_date}")

# Each data structure solves different problems - choose based on your needs

# ============================
# 🏗️ Advanced Collections
# ============================

# Python's collections module provides specialized data structures
# These solve common patterns more elegantly than basic lists and dicts

print("\n🏗️ ADVANCED COLLECTIONS")
print("-" * 23)

# === Counter: Automatic frequency counting ===
print("🔢 Counter - Automatic counting:")
feedback_words = [
    "good",
    "excellent",
    "poor",
    "good",
    "average",
    "excellent",
    "excellent",
    "good",
]
word_counts = Counter(feedback_words)
print(f"   Feedback frequency: {dict(word_counts)}")
print(f"   Most common feedback: {word_counts.most_common(2)}")

# === defaultdict: Dictionaries with automatic default values ===
print("\n📊 defaultdict - Automatic grouping:")
# Creates missing keys with default values automatically - no KeyError!
student_grades = [
    ("Alice", "Math", 92),
    ("Bob", "Math", 87),
    ("Alice", "Science", 95),
    ("Charlie", "Math", 78),
    ("Bob", "Science", 91),
]

# Group grades by student using defaultdict
grades_by_student = defaultdict(list)  # Creates empty list for new keys
for student, subject, grade in student_grades:
    grades_by_student[student].append((subject, grade))

for student, grades in grades_by_student.items():
    avg_grade = sum(grade for _, grade in grades) / len(grades)
    print(f"   {student}: {grades} → Average: {avg_grade:.1f}")

# === namedtuple: Structured data with named fields ===
print("\n📋 namedtuple - Structured data:")
# Like a lightweight class - creates objects with named fields
# More readable than regular tuples, but still immutable
Employee = namedtuple("Employee", ["name", "department", "salary", "years"])

employees = [
    Employee("Alice", "Engineering", 95000, 3),
    Employee("Bob", "Marketing", 70000, 2),
    Employee("Charlie", "Engineering", 105000, 5),
]

print("   Employee database:")
for emp in employees:
    print(f"   {emp.name} ({emp.department}): ${emp.salary:,} - {emp.years} years")

# Calculate department statistics using structured data
eng_salaries = [emp.salary for emp in employees if emp.department == "Engineering"]
avg_eng_salary = sum(eng_salaries) / len(eng_salaries)
print(f"   Engineering average salary: ${avg_eng_salary:,.0f}")

# Advanced collections make your code more expressive and less error-prone

# ============================
# 🧮 Algorithms and Problem Solving
# ============================

# Algorithms are step-by-step procedures for solving problems efficiently
# In data science, we use algorithms for sorting, searching, and analysis
# Understanding core algorithms makes you a better problem solver

print("\n🧮 ALGORITHMS AND PROBLEM SOLVING")
print("-" * 33)

# === Sorting: Organizing data by criteria ===
print("🔄 Sorting algorithms:")

# Sample data: customer records for analysis
customers = [
    {"name": "Alice", "age": 28, "purchases": 15, "total_spent": 1500},
    {"name": "Bob", "age": 35, "purchases": 8, "total_spent": 800},
    {"name": "Charlie", "age": 22, "purchases": 25, "total_spent": 2200},
    {"name": "Diana", "age": 31, "purchases": 12, "total_spent": 1100},
]

# Sort by different criteria using key functions
by_age = sorted(customers, key=lambda x: x["age"])
by_spending = sorted(customers, key=lambda x: x["total_spent"], reverse=True)
by_frequency = sorted(customers, key=lambda x: x["purchases"], reverse=True)

print("   Sorted by age:")
for customer in by_age:
    print(f"     {customer['name']} ({customer['age']} years old)")

print("   Top spenders:")
for customer in by_spending[:2]:  # Top 2 only
    print(f"     {customer['name']}: ${customer['total_spent']:,}")

# === Binary Search: Efficient searching ===
print("\n🔍 Binary Search - Efficient searching:")


def binary_search(sorted_list: List[int], target: int) -> int:
    """
    🎯 Find target in sorted list using binary search
    Much faster than checking every item (especially for large datasets)!

    Time complexity: O(log n) instead of O(n) - huge difference for big data
    """
    left, right = 0, len(sorted_list) - 1
    steps = 0

    while left <= right:
        steps += 1
        mid = (left + right) // 2
        mid_value = sorted_list[mid]

        print(f"     Step {steps}: checking position {mid} (value {mid_value})")

        if mid_value == target:
            print(f"     ✅ Found {target} at position {mid} in {steps} steps!")
            return mid
        elif mid_value < target:
            left = mid + 1  # Search right half
        else:
            right = mid - 1  # Search left half

    print(f"     ❌ {target} not found after {steps} steps")
    return -1


# Test binary search efficiency
sorted_numbers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
print(f"   Searching for 15 in: {sorted_numbers}")
binary_search(sorted_numbers, 15)

# === Practical Algorithm: Duplicate Detection ===
print("\n🔍 Practical: Finding duplicates in data:")


def analyze_duplicates(data: List[Any]) -> Dict[str, Any]:
    """
    🧹 Analyze data for duplicate entries - essential for data quality!
    Uses set for O(1) lookups instead of O(n) list searches
    """
    seen = set()
    duplicates = []

    for item in data:
        if item in seen:
            duplicates.append(item)
        else:
            seen.add(item)

    unique_duplicates = list(set(duplicates))

    return {
        "original_count": len(data),
        "unique_count": len(seen),
        "duplicate_items": unique_duplicates,
        "duplicate_count": len(duplicates),
        "is_clean": len(duplicates) == 0,
    }


# Test with realistic messy customer data
customer_ids = [101, 102, 103, 102, 104, 105, 103, 106, 101, 107]
duplicate_analysis = analyze_duplicates(customer_ids)

print("   Customer ID analysis:")
print(f"     Original data: {customer_ids}")
print(f"     Total records: {duplicate_analysis['original_count']}")
print(f"     Unique customers: {duplicate_analysis['unique_count']}")
print(f"     Duplicate IDs: {duplicate_analysis['duplicate_items']}")
print(f"     Data is clean: {duplicate_analysis['is_clean']}")

# Algorithms help you process data efficiently and solve complex problems systematically

# ============================
# 🎯 Putting It All Together
# ============================

# Real-world example combining all the concepts we've learned
# This shows how everything connects in a complete data analysis pipeline

print("\n🎯 PUTTING IT ALL TOGETHER")
print("-" * 28)


def analyze_customer_data(customers_raw: List[str]) -> Dict[str, Any]:
    """
    📊 Complete data analysis pipeline combining our key concepts:
    - String processing (data cleaning)
    - Data structures (organizing information)
    - Control flow (handling different scenarios)
    - Error handling (robust processing)
    - Algorithms (efficient analysis)
    """

    # Initialize our data containers
    customers = []
    errors = []

    # Process each raw customer record
    for i, line in enumerate(customers_raw, 1):
        try:
            # Clean the input data
            line = line.strip()
            if not line:
                continue

            # Parse customer info (expecting: name,age,email,purchases)
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                errors.append(f"Line {i}: Wrong number of fields")
                continue

            name, age_str, email, purchases_str = parts

            # Validate and convert data types
            age = int(age_str)
            purchases = int(purchases_str)

            # Business logic validation
            if age < 0 or age > 120:
                errors.append(f"Line {i}: Invalid age {age}")
                continue

            if purchases < 0:
                errors.append(f"Line {i}: Invalid purchases {purchases}")
                continue

            # Store clean, validated data
            customers.append(
                {
                    "name": name.title(),  # Standardize capitalization
                    "age": age,
                    "email": email.lower(),  # Standardize email format
                    "purchases": purchases,
                }
            )

        except ValueError as e:
            errors.append(f"Line {i}: {e}")
        except Exception as e:
            errors.append(f"Line {i}: Unexpected error - {e}")

    # Handle case where no valid data was found
    if not customers:
        return {"error": "No valid customer data found", "errors": errors}

    # Analyze the clean data
    total_customers = len(customers)
    total_purchases = sum(c["purchases"] for c in customers)
    avg_purchases = total_purchases / total_customers

    # Categorize customers by age groups
    age_groups = defaultdict(int)
    for customer in customers:
        if customer["age"] < 25:
            age_groups["Young (18-24)"] += 1
        elif customer["age"] < 35:
            age_groups["Adult (25-34)"] += 1
        elif customer["age"] < 50:
            age_groups["Middle-aged (35-49)"] += 1
        else:
            age_groups["Senior (50+)"] += 1

    # Find top customers by purchase activity
    top_customers = sorted(customers, key=lambda x: x["purchases"], reverse=True)[:3]

    return {
        "total_customers": total_customers,
        "total_purchases": total_purchases,
        "average_purchases": round(avg_purchases, 1),
        "age_distribution": dict(age_groups),
        "top_customers": [(c["name"], c["purchases"]) for c in top_customers],
        "errors": errors,
        "data_quality": f"{len(errors)} errors in {len(customers_raw)} records",
    }


# Test our complete analysis with realistic messy data
raw_customer_data = [
    "Alice Johnson, 28, alice@email.com, 15",
    "Bob Smith, 35, bob@email.com, 8",
    "Charlie Brown, invalid_age, charlie@email.com, 25",  # Error: invalid age
    "Diana Wilson, 31, diana@email.com, 12",
    "Eve Davis, 45, eve@email.com, 30",
    ", 25, incomplete@email.com, 5",  # Error: missing name
    "Frank Miller, 29, frank@email.com, 18",
]

print("📈 Complete Customer Analysis:")
results = analyze_customer_data(raw_customer_data)

for key, value in results.items():
    if key == "top_customers":
        print(f"   {key}:")
        for name, purchases in value:
            print(f"     {name}: {purchases} purchases")
    elif key == "errors":
        if value:  # Only show if there are errors
            print(f"   {key}: {value}")
    else:
        print(f"   {key}: {value}")

print("\n" + "=" * 50)
print("🎉 Tutorial complete! You've mastered Python fundamentals! 🚀")

"""
🎯 COMPREHENSIVE KEY TAKEAWAYS
=============================

✅ Variables & Types
   - Python uses dynamic typing - it figures out data types automatically
   - Core types: str (text), int (whole numbers), float (decimals), bool (True/False)
   - Use type() to check what kind of data you're working with

✅ Strings & Text Processing
   - Essential for cleaning messy real-world data
   - Key methods: .strip(), .lower(), .title(), .split()
   - f-strings are the modern way to format text: f"Hello {name}!"

✅ Conditional Logic (if/elif/else)
   - Let programs make decisions based on data conditions
   - Use comparison operators: ==, !=, <, >, <=, >=
   - Essential for data validation and business logic

✅ Loops & Iteration
   - for loops process collections of data efficiently
   - enumerate() gives you both position and item
   - zip() combines multiple lists element-by-element
   - List comprehensions are Python's superpower for data transformation

✅ Functions
   - Organize code into reusable, testable blocks
   - Use type hints to document what functions expect and return
   - Default parameters make functions flexible
   - Return dictionaries for complex results

✅ Data Structures
   - Lists: Ordered, changeable [1, 2, 3]
   - Dictionaries: Key-value pairs {"name": "Alice", "age": 28}
   - Sets: Unique items only {1, 2, 3} - great for removing duplicates
   - Tuples: Immutable sequences (1, 2, 3) - good for coordinates, records

✅ Advanced Collections
   - Counter: Automatic frequency counting
   - defaultdict: Dictionaries with automatic default values
   - namedtuple: Structured data with named fields

✅ Error Handling
   - try/except prevents crashes and handles problems gracefully
   - Always anticipate what could go wrong with your data
   - Return None or default values for recoverable errors

✅ Algorithms & Problem Solving
   - Sorting: Use sorted() with key functions for custom criteria
   - Binary search: O(log n) efficiency for searching sorted data
   - Set operations: Fast duplicate detection and data comparison
   - Choose algorithms based on data size and performance needs

🚀 Why This Matters for Data Science:
   - These concepts are the building blocks for pandas, NumPy, scikit-learn
   - Every data pipeline involves cleaning (strings), filtering (conditionals),
     processing (loops), and organizing (data structures)
   - Functions make your analysis reproducible and testable
   - Error handling prevents crashes when data is messy or incomplete

🎉 Congratulations! You've just walked through Python's core building blocks.
   You're now ready to tackle real-world data problems with confidence!

Next up: Move on to data manipulation with pandas and NumPy! 📊
"""
