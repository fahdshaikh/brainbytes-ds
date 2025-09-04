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
You'll only need a handful of standard Python packages to run this and all other tutorials in the minicourse:
NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, and SciPy.

If you're setting this up from scratch, just run:

    pip install -r requirements.txt

🤖 Note:
This is part of the [Cheatsheet DS Minicourse] — a side project made with
some coding, some love, and some help from Claude AI.

Let's dive in! 🎯
"""


# -----------------------------------------------------------------------------
# 📦 Imports and Setup
# -----------------------------------------------------------------------------

# 💡 What are imports and why do we need them?
# Think of imports like borrowing tools from a toolbox. Python comes with tons of built-in
# modules (like pre-made tools) that add extra functionality to your program. Instead of
# writing everything from scratch, we can use these ready-made tools!

# 🔧 Why import at the top?
# It's like laying out all your tools before starting work. This way:
# 1. Anyone reading your code knows what tools you're using
# 2. All tools are available throughout your entire program
# 3. If there's a missing tool, you'll know immediately when the program starts

import random  # 🎲 For generating random numbers - super useful for testing!
from collections import (
    Counter,
    defaultdict,
    namedtuple,
)  # 📚 Special data containers that make life easier
from typing import (
    List,
    Dict,
    Optional,
    Any,
)  # 📝 For documenting what types of data our functions expect

# 🎯 Why set a random seed?
# When we use "random" functions, we want our examples to be reproducible.
# Setting seed(42) means every time you run this code, you'll get the same "random" results.
# This is CRUCIAL in data science - you want your analysis to be repeatable!
random.seed(42)

print("🚀 Welcome to Python Fundamentals!")
print("=" * 40)

# -----------------------------------------------------------------------------
# 📌 Variables and Data Types
# -----------------------------------------------------------------------------

# 🤔 What exactly is a variable?
# A variable is like a labeled box where you store information. Instead of remembering
# that the number 28 represents someone's age, you can put it in a box labeled "age".
# This makes your code readable and reusable!

# 🌟 What makes Python special? DYNAMIC TYPING!
# In languages like Java or C++, you have to say "this will store a number" before using it.
# Python is smart - it looks at what you're storing and figures out the type automatically.
# This makes Python super beginner-friendly and fast to write!

print("\n📌 VARIABLES AND DATA TYPES")
print("-" * 28)

# 📦 Python's core data types - these are the building blocks of EVERYTHING!
# Each type is designed for different kinds of information:

name = (
    "Alice"  # 🔤 str (string) - ANY text data (letters, symbols, even numbers as text!)
)
# Why strings? Because most real-world data is text: names, addresses, categories, etc.

age = 28  # 🔢 int (integer) - whole numbers only (no decimals)
# Perfect for counting things: people, items, days, etc.

salary = 75000.50  # 💰 float - numbers with decimal points
# Essential for money, measurements, percentages, scientific data

is_employed = True  # ✅ bool (boolean) - only True or False values
# Great for yes/no questions, flags, conditions

# 🎨 f-strings: Python's modern, clean way to format text
# The 'f' stands for "formatted string". Put variables inside {} and they get inserted!
# This is WAY cleaner than old methods like "Hello " + name + "!"
print(f"👋 Meet {name}:")
print(f"   Age: {age} years")
print(f"   Salary: ${salary:,.2f}")  # 💡 :,.2f = add commas + show 2 decimal places
print(f"   Employed: {is_employed}")

# 🔍 The type() function - your debugging best friend!
# Ever wonder "what kind of data am I actually working with?" type() tells you!
# This is SUPER useful when things go wrong or when working with messy real-world data
print("\n🔍 Data types:")
print(f"   '{name}' is a {type(name).__name__}")  # __name__ just makes it look cleaner
print(f"   {age} is an {type(age).__name__}")
print(f"   {salary} is a {type(salary).__name__}")
print(f"   {is_employed} is a {type(is_employed).__name__}")

# 🚀 Key insight about Python's dynamic typing:
# You don't declare types upfront like "int age = 28". Python figures it out!
# This makes Python code shorter and more flexible, but sometimes less predictable.

# -----------------------------------------------------------------------------
# 🧵 Strings and Text Processing
# -----------------------------------------------------------------------------

# 🌍 Why are strings SO important in data science?
# Real-world data is messy! Customer names come in as "john", "JOHN", "  John  ".
# Addresses, categories, survey responses - they all need cleaning and standardization.
# String methods are your data-cleaning superpowers!

# 📊 Common real-world string problems:
# - Inconsistent capitalization (john vs JOHN vs John)
# - Extra whitespace ("  Data Corp  " vs "Data Corp")
# - Different formats for the same thing
# String methods help us fix ALL of these issues!

print("\n🧵 STRINGS AND TEXT PROCESSING")
print("-" * 32)

# 📝 Realistic messy data that needs cleaning (this happens ALL THE TIME!)
first_name = "john"  # ❌ Should be capitalized
last_name = "DOE"  # ❌ All caps, should be normal case
company = "  Data Corp  "  # ❌ Extra spaces (happens when copy-pasting from forms)

# 🧹 String methods for cleaning and formatting - your cleaning toolkit!
full_name = (
    first_name.title() + " " + last_name.title()
)  # 🎩 .title() capitalizes the first letter of each word (proper names!)

# 💡 Why .lower() for emails? Email addresses are case-insensitive,
# so "John.Doe@email.com" and "john.doe@email.com" are the same.
# Using .lower() standardizes everything to avoid duplicates!
email = f"{first_name.lower()}.{last_name.lower()}@datacorp.com"

clean_company = company.strip()  # ✂️ .strip() removes whitespace from both ends

print("📝 String cleaning in action:")
print(f"   Raw data: '{first_name}', '{last_name}', '{company}'")
print(f"   Clean name: {full_name}")
print(f"   Generated email: {email}")
print(f"   Clean company: '{clean_company}'")

# 🛠️ Essential string methods you'll use CONSTANTLY in data science
sample_text = "  Machine Learning is AWESOME!  "
print("\n🧹 Common text cleaning operations:")
print(f"   Original: '{sample_text}'")
print(f"   Stripped: '{sample_text.strip()}'")  # Remove leading/trailing whitespace
print(f"   Lowercase: '{sample_text.strip().lower()}'")  # Standardize to lowercase
print(
    f"   Split into words: {sample_text.strip().split()}"
)  # Break into individual words (list)
print(f"   Character count: {len(sample_text.strip())}")  # How long is the clean text?

# 🎯 Why this matters for real projects:
# Almost every data science project involves string manipulation!
# - Cleaning customer names and addresses
# - Standardizing product categories
# - Processing survey responses
# - Parsing file names and URLs
# Master these methods and you'll save HOURS of work later!

# -----------------------------------------------------------------------------
# 🔀 Conditional Logic (if/elif/else)
# -----------------------------------------------------------------------------

# 🤖 Why do programs need to make decisions?
# Just like humans, programs need to handle different situations differently.
# "IF it's raining, take an umbrella. ELSE, don't."
# "IF customer spent > $1000, give them VIP status. ELSE, regular status."

# 🎯 Conditionals are the foundation of:
# - Data validation ("Is this email address valid?")
# - Business logic ("Which discount should this customer get?")
# - Data filtering ("Show only customers from California")
# - Error handling ("Is this data in the expected format?")

print("\n🔀 CONDITIONAL LOGIC")
print("-" * 20)


# 💡 Functions preview! We're defining a reusable piece of code.
# Think of it like creating a custom tool that you can use over and over.
def categorize_score(score):
    """
    🎯 Categorize a test score into letter grades

    This demonstrates several key concepts:
    - if/elif/else structure for handling multiple conditions
    - Comparison operators (>=, <, etc.) for making decisions
    - How to return different values based on logic
    - Real-world application (grade calculation)

    💡 Why use elif instead of multiple if statements?
    elif means "else if" - it only checks if previous conditions were False.
    This is more efficient and prevents conflicting conditions!
    """

    # 🎲 Decision tree: check conditions from highest to lowest score
    if score >= 90:  # First condition: excellent performance
        return f"A ({score}%) - Excellent!"
    elif score >= 80:  # Only checks this if score < 90
        return f"B ({score}%) - Good work!"
    elif score >= 70:  # Only checks this if score < 80
        return f"C ({score}%) - Passing"
    elif score >= 60:  # Only checks this if score < 70
        return f"D ({score}%) - Needs improvement"
    else:  # If none of the above conditions are true
        return f"F ({score}%) - Failing"


# 🧪 Test our conditional logic with various scores
# This shows how the same function handles different inputs differently!
test_scores = [95, 87, 72, 65, 45]
print("📊 Score categorization:")
for (
    score
) in test_scores:  # 🔄 Loop through each score (we'll learn more about loops soon!)
    result = categorize_score(score)
    print(f"   {result}")

# 🌟 Key takeaway: Conditionals make your programs smart!
# They let you handle different data scenarios automatically, which is essential
# when working with real-world data that varies widely.

# 🎯 Common comparison operators you'll use:
# == (equal to), != (not equal to), < (less than), > (greater than),
# <= (less than or equal), >= (greater than or equal)
# and (both conditions true), or (either condition true), not (opposite)

# -----------------------------------------------------------------------------
# 🔄 Loops and Iteration
# -----------------------------------------------------------------------------

# 🤔 Why do we need loops? What problem do they solve?
# Imagine you have 1000 customer records and need to process each one.
# Without loops, you'd need to copy-paste the same code 1000 times! 😱
# Loops let us say "do this action for every item in my collection" efficiently.

# 🚀 In data science, loops are EVERYWHERE:
# - Processing datasets row by row
# - Applying transformations to multiple columns
# - Generating reports for different time periods
# - Running experiments with different parameters

print("\n🔄 LOOPS AND ITERATION")
print("-" * 22)

# 📊 Sample data: product inventory (simulating real business data)
# In real life, this might come from a database or CSV file
products = ["laptop", "mouse", "keyboard", "monitor", "webcam"]
prices = [999, 25, 75, 300, 120]
stock_levels = [15, 50, 30, 8, 25]

print("📦 Inventory Report:")
print("   ID | Product     | Price | Stock | Status")
print("   ---|-------------|-------|-------|--------")

# 🎯 The magic combo: enumerate() + zip()
# enumerate() gives us both position (index) and item from a list
# zip() combines multiple lists element-by-element
# This lets us process related data together efficiently!

# 💡 Why start enumerate at 1?
# enumerate(list, 1) starts counting from 1 instead of 0, which is more natural for IDs
for i, (product, price, stock) in enumerate(zip(products, prices, stock_levels), 1):
    # 🧠 Business logic: determine stock status based on inventory levels
    # This is conditional logic inside a loop - super common pattern!
    if stock < 10:  # Low inventory - need to reorder soon!
        status = "LOW"
    elif stock < 20:  # Okay for now, but watch it
        status = "OK"
    else:  # Well stocked
        status = "GOOD"

    # 🎨 Format output in a clean table
    # {i:2d} = integer with 2 characters width, {product:11s} = string with 11 characters width
    print(f"   {i:2d} | {product:11s} | ${price:4d} | {stock:5d} | {status}")

# 🚀 List comprehensions: Python's superpower for transforming data!
# These are like loops but more concise and often faster
# Think of them as "create a new list by doing something to each item"

print("\n🚀 List comprehensions (Python's superpower):")

# 🎯 Pattern: [what_to_do for item in collection if condition]
# This is SO much cleaner than writing full for loops for simple transformations!

# Filter expensive products using comprehension
expensive_products = [
    product for product, price in zip(products, prices) if price > 100
]  # "Give me each product where price > 100"
print(f"   Expensive items: {expensive_products}")

# Calculate inventory values (price × stock) for each product
total_values = [price * stock for price, stock in zip(prices, stock_levels)]
# "For each product, multiply price by stock level"
print(f"   Inventory values: {total_values}")

# Find products that need restocking
low_stock_items = [
    product for product, stock in zip(products, stock_levels) if stock < 15
]  # "Give me products where stock < 15"
print(f"   Need restocking: {low_stock_items}")

# 🌟 Why list comprehensions are awesome:
# 1. More readable once you get used to them
# 2. Often faster than regular loops
# 3. Less code = fewer bugs
# 4. Very "Pythonic" (idiomatic Python style)

# 🎯 Key insight: Loops are everywhere in data processing!
# Master loops and comprehensions, and you'll be able to automate almost anything!

# -----------------------------------------------------------------------------
# 🧠 Functions
# -----------------------------------------------------------------------------

# 🤔 What exactly is a function and why do we need them?
# Think of a function like a recipe or a reusable tool:
# - INPUT: ingredients (parameters)
# - PROCESS: cooking steps (your code)
# - OUTPUT: finished dish (return value)

# 🎯 Functions solve several problems:
# 1. REUSABILITY: Write once, use many times
# 2. ORGANIZATION: Break complex problems into smaller pieces
# 3. TESTING: Easy to test individual parts of your program
# 4. COLLABORATION: Team members can work on different functions
# 5. DEBUGGING: Easier to find and fix problems

print("\n🧠 FUNCTIONS")
print("-" * 12)

# 💡 Type hints explanation (those List[float], Dict[str, Any] things):
# These are like documentation for your function - they tell others (and yourself!)
# what type of data your function expects and returns.
# Python doesn't enforce them, but they make your code much more readable!


def analyze_sales(daily_sales: List[float], target: float = 1000.0) -> Dict[str, Any]:
    """
    📊 Analyze daily sales performance and return insights

    Parameters:
        daily_sales: List of daily sales amounts (why List[float]? We expect multiple decimal numbers!)
        target: Target sales amount (defaults to 1000.0 - this is a DEFAULT PARAMETER!)

    Returns:
        Dictionary containing analysis results (why Dict[str, Any]? Keys are strings, values can be anything!)

    🎯 What are default parameters?
    target=1000.0 means if someone doesn't provide a target, we'll use 1000.0 automatically.
    This makes functions more flexible and easier to use!
    """

    # 🧮 Calculations using built-in functions:
    total_sales = sum(daily_sales)  # sum() adds all numbers in a list
    avg_sales = total_sales / len(daily_sales)  # len() gives us the count of items
    best_day = max(daily_sales)  # max() finds the highest value
    worst_day = min(daily_sales)  # min() finds the lowest value

    # 🔍 This line uses a LIST COMPREHENSION with a CONDITION:
    # "Count how many days had sales > target"
    # sum(1 for ...) is a clever trick to count items that meet a condition
    days_above_target = sum(1 for sales in daily_sales if sales > target)

    # 📦 Return a dictionary with all our calculated insights
    # Why a dictionary? It's like a labeled box for each piece of information!
    return {
        "total": total_sales,
        "average": avg_sales,
        "best_day": best_day,
        "worst_day": worst_day,
        "target": target,
        "days_above_target": days_above_target,
        "success_rate": (days_above_target / len(daily_sales))
        * 100,  # Percentage calculation
    }


# 🧪 Test our function with realistic sales data
# This shows how functions make your code reusable - same function, different data!
weekly_sales = [850, 1200, 950, 1100, 800, 1300, 900]
analysis = analyze_sales(weekly_sales, target=1000)  # Using our custom target

print("📈 Sales Analysis Results:")
# 🔄 Loop through dictionary items (key-value pairs)
for key, value in analysis.items():
    # 🎯 Check if value is a float to format it nicely
    if isinstance(value, float):  # isinstance() checks the type of data
        print(f"   {key}: {value:.2f}")  # .2f means 2 decimal places
    else:
        print(f"   {key}: {value}")

# 🌟 Key benefits we just demonstrated:
# 1. Our function is REUSABLE - we can analyze any sales data
# 2. It's TESTABLE - easy to verify it works correctly
# 3. It's ORGANIZED - all sales analysis logic is in one place
# 4. It's DOCUMENTED - clear what it does and expects

# 🎯 Functions are the building blocks of larger programs!
# Every complex data analysis is just many functions working together.

# -----------------------------------------------------------------------------
# 🛡️ Error Handling
# -----------------------------------------------------------------------------

# 🤔 Why do we need error handling? What could go wrong?
# Real world is messy! Users make mistakes, files get corrupted, networks fail.
# Without error handling, one bad piece of data crashes your entire program! 💥

# 🎯 Common problems in data science:
# - Missing files ("file not found")
# - Corrupted data (letters where numbers should be)
# - Network issues (API calls fail)
# - Invalid user input (negative ages, empty fields)
# - Division by zero errors

# 🛡️ Error handling lets us:
# 1. CONTINUE RUNNING even when things go wrong
# 2. PROVIDE HELPFUL MESSAGES instead of cryptic errors
# 3. RECOVER GRACEFULLY by using default values or skipping bad data
# 4. LOG PROBLEMS for later investigation

print("\n🛡️ ERROR HANDLING")
print("-" * 17)


# 💡 Optional[float] means "either a float number OR None"
# This clearly communicates that our function might not return a number if something goes wrong
def safe_calculate(operation: str, a: float, b: float) -> Optional[float]:
    """
    🔒 Perform calculations safely without crashing the program

    Returns the result if successful, None if something goes wrong

    🎯 Why return None for errors?
    None is Python's way of saying "no value" or "nothing here".
    It's better than crashing or returning a weird number like -999!
    """

    # 🛡️ try/except block: "TRY to do this, but if something goes wrong, do that instead"
    try:
        # 🧮 Handle different types of calculations
        if operation == "divide":
            # 🚨 Check for division by zero BEFORE doing the calculation
            # This prevents the dreaded "ZeroDivisionError"
            if b == 0:
                print("   ⚠️  Cannot divide by zero!")
                return None  # Return "no value" instead of crashing

            result = a / b

        elif operation == "square_root":
            # 🚨 Check for negative numbers (can't take square root of negative in real numbers)
            if a < 0:
                print("   ⚠️  Cannot take square root of negative number!")
                return None

            result = a**0.5  # ** means "to the power of" (0.5 = square root)

        else:
            # 🚨 Handle unknown operations gracefully
            print(f"   ⚠️  Unknown operation: {operation}")
            return None

        # ✅ If we get here, everything worked fine!
        print(f"   ✅ {operation}({a}, {b}) = {result:.2f}")
        return result

    # 🎯 Specific error types for specific problems:
    except TypeError:
        # This happens when someone passes wrong data types (like text instead of numbers)
        print(f"   ❌ Invalid input types: {type(a)}, {type(b)}")
        return None

    except Exception as e:
        # This catches ANY other unexpected error
        # 'as e' lets us see what the error actually was
        print(f"   ❌ Unexpected error: {e}")
        return None


# 🧪 Test our error handling with various scenarios
# This shows how robust programs handle ALL kinds of input, not just the happy path!
print("🧪 Testing error handling:")
safe_calculate("divide", 10, 2)  # ✅ Should work fine
safe_calculate("divide", 10, 0)  # ❌ Division by zero
safe_calculate("square_root", -4, 0)  # ❌ Negative square root
safe_calculate("unknown", 5, 3)  # ❌ Unknown operation

# 🌟 What did we just accomplish?
# Our program handled 4 different scenarios without crashing once!
# In real data science projects, this kind of robustness is ESSENTIAL.

# 🎯 Key principles of good error handling:
# 1. ANTICIPATE what could go wrong
# 2. HANDLE errors gracefully (don't crash!)
# 3. PROVIDE helpful error messages
# 4. RETURN reasonable values (like None) for errors
# 5. LOG what went wrong for debugging later

# -----------------------------------------------------------------------------
# 📚 Data Structures
# -----------------------------------------------------------------------------

# 🤔 What are data structures and why do we need different ones?
# Data structures are containers that organize and store information in different ways.
# Just like you wouldn't use a shopping bag to store soup, different data needs different containers!

# 🎯 Each structure is optimized for different operations:
# - Lists: Great for ordered data you need to change
# - Dictionaries: Super fast lookups by key (like phone books)
# - Sets: Automatically remove duplicates, fast membership testing
# - Tuples: Immutable data that won't accidentally change

print("\n📚 DATA STRUCTURES")
print("-" * 18)

# === LISTS: Your Swiss Army Knife for Ordered Data ===
print("📋 Lists - Ordered and changeable:")

# 💡 Why use range(1, 11)?
# range() creates a sequence of numbers. range(1, 11) gives us 1, 2, 3, ..., 10
# (The end number 11 is NOT included - this is a Python quirk!)
numbers = list(range(1, 11))  # Creates [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(f"   Original: {numbers}")

# 🛠️ Common list operations you'll use frequently:
numbers.append(11)  # Add to the END of the list
numbers.insert(0, 0)  # Insert at position 0 (the beginning)
numbers.remove(5)  # Remove the FIRST occurrence of the value 5

print(f"   Modified: {numbers}")

# 🚀 List comprehensions for data transformation (very Pythonic!)
# These are like saying "give me a new list where I do something to each item"

evens = [n for n in numbers if n % 2 == 0]  # % is "modulo" - remainder after division
# "Give me all numbers where remainder when divided by 2 equals 0" (even numbers!)

squares = [n**2 for n in numbers[:5]]  # [:5] means "first 5 elements"
# "Give me the square of each of the first 5 numbers"

print(f"   Even numbers: {evens}")
print(f"   First 5 squared: {squares}")

# === DICTIONARIES: Fast Lookups by Key ===
print("\n📖 Dictionaries - Key-value mappings:")

# 🎯 Think of dictionaries like a real dictionary or phone book:
# You look up a WORD (key) to find its DEFINITION (value)
# You look up a NAME (key) to find a PHONE NUMBER (value)
# This makes dictionaries SUPER fast for lookups!

sales_data = {"January": 15000, "February": 18000, "March": 12000, "April": 22000}
print(f"   Monthly sales: {sales_data}")

# 🛠️ Dictionary operations for analysis:
sales_data["May"] = (
    25000  # Add new key-value pair (like adding a new page to the phone book)
)

# .values() gets all the values (ignoring the keys)
total_sales = sum(sales_data.values())

# 🎯 This is a cool trick! max() with key parameter:
# "Find the key that has the maximum value"
best_month = max(sales_data, key=sales_data.get)

print(f"   Total sales: ${total_sales:,}")  # :, adds thousand separators
print(f"   Best month: {best_month} (${sales_data[best_month]:,})")

# 🚀 Dictionary comprehensions for filtering data:
# "Give me a new dictionary with only months where sales > 15000"
high_performing = {
    month: amount for month, amount in sales_data.items() if amount > 15000
}
print(f"   High performers: {high_performing}")

# === SETS: Automatic Duplicate Removal ===
print("\n🎯 Sets - Collections of unique items:")

# 🎯 Sets automatically remove duplicates - super useful for data cleaning!
# Notice how "python" and "sql" appear twice, but sets keep only one copy
team_skills = {"python", "sql", "excel", "tableau", "python", "sql"}  # duplicates!
print(f"   Team skills (duplicates removed): {team_skills}")

required_skills = {"python", "sql", "statistics"}
nice_to_have = {"tableau", "r", "machine_learning"}

# 🧮 Set operations for comparing groups (like Venn diagrams!):
print(f"   Required skills: {required_skills}")

# & means "intersection" - what's common between both sets
print(f"   Skills we have: {team_skills & required_skills}")

# - means "difference" - what's in first set but not in second
print(f"   Skills we're missing: {required_skills - team_skills}")

# | means "union" - combine all unique items from all sets
print(f"   All possible skills: {team_skills | required_skills | nice_to_have}")

# === TUPLES: Immutable Sequences ===
print("\n📌 Tuples - Unchangeable sequences:")

# 🔒 Why use tuples instead of lists?
# Tuples can't be changed after creation (immutable). This is good for:
# - Coordinates (x, y) that shouldn't accidentally change
# - Database records that represent fixed data
# - Function returns where you want to return multiple values together

employee_record = ("Alice Johnson", "Data Analyst", 75000, "2022-01-15")

# 📦 Tuple unpacking: extracting values into separate variables
# This is like opening a package and taking out each item
name, position, salary, start_date = employee_record

print(f"   Employee: {name}, {position}, ${salary:,}, started {start_date}")

# 🌟 Key takeaway: Choose your data structure based on what you need to do!
# - Need to change items? → List
# - Need fast lookups by key? → Dictionary
# - Need to remove duplicates? → Set
# - Need unchangeable data? → Tuple

# -----------------------------------------------------------------------------
# 🏗️ Advanced Collections
# -----------------------------------------------------------------------------

# 🤔 Why do we need "advanced" collections when we have lists and dicts?
# Python's collections module provides specialized data structures that solve
# common patterns more elegantly and efficiently than basic structures.
# They're like power tools - designed for specific jobs!

print("\n🏗️ ADVANCED COLLECTIONS")
print("-" * 23)

# === COUNTER: Automatic Frequency Counting ===
print("🔢 Counter - Automatic counting:")

# 🎯 Counter solves a super common problem: "How many times does each item appear?"
# Without Counter, you'd need to write a loop and manually count everything.
# Counter does this automatically!

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

# ✨ Magic! Counter automatically counts each word's frequency
word_counts = Counter(feedback_words)
print(
    f"   Feedback frequency: {dict(word_counts)}"
)  # Convert to dict for pretty printing

# 🏆 .most_common() gives you the top items - super useful for analysis!
print(f"   Most common feedback: {word_counts.most_common(2)}")  # Top 2 most frequent

# === DEFAULTDICT: Dictionaries with Automatic Default Values ===
print("\n📊 defaultdict - Automatic grouping:")

# 🤔 What problem does defaultdict solve?
# Regular dictionaries give you a KeyError if you try to access a key that doesn't exist.
# defaultdict automatically creates missing keys with a default value!
# This is PERFECT for grouping data without checking if keys exist first.

student_grades = [
    ("Alice", "Math", 92),
    ("Bob", "Math", 87),
    ("Alice", "Science", 95),
    ("Charlie", "Math", 78),
    ("Bob", "Science", 91),
]

# 🎯 Group grades by student using defaultdict
# defaultdict(list) means "if a key doesn't exist, create it with an empty list"
grades_by_student = defaultdict(list)

# 🔄 Process each grade record
for student, subject, grade in student_grades:
    # ✨ Magic! If student doesn't exist as a key, defaultdict creates it automatically
    grades_by_student[student].append((subject, grade))

# 📊 Calculate statistics for each student
for student, grades in grades_by_student.items():
    # 🧮 Calculate average: sum all grades, divide by count
    avg_grade = sum(grade for _, grade in grades) / len(grades)
    print(f"   {student}: {grades} → Average: {avg_grade:.1f}")

# === NAMEDTUPLE: Structured Data with Named Fields ===
print("\n📋 namedtuple - Structured data:")

# 🤔 What's the problem with regular tuples?
# employee = ("Alice", "Engineering", 95000, 3)  # What does each number mean??
# namedtuple fixes this by giving names to each position!

# 🏗️ Create a custom data structure (like a lightweight class)
# namedtuple is like creating a template for structured data
Employee = namedtuple("Employee", ["name", "department", "salary", "years"])

# 📝 Create employee records using our template
employees = [
    Employee("Alice", "Engineering", 95000, 3),  # Much clearer than unnamed tuple!
    Employee("Bob", "Marketing", 70000, 2),
    Employee("Charlie", "Engineering", 105000, 5),
]

print("   Employee database:")
for emp in employees:
    # 🎯 Access fields by name instead of position - much more readable!
    print(f"   {emp.name} ({emp.department}): ${emp.salary:,} - {emp.years} years")

# 🧮 Calculate department statistics using structured data
eng_salaries = [emp.salary for emp in employees if emp.department == "Engineering"]
avg_eng_salary = sum(eng_salaries) / len(eng_salaries)
print(f"   Engineering average salary: ${avg_eng_salary:,.0f}")

# 🌟 Why advanced collections are awesome:
# 1. Counter: Automatic frequency analysis
# 2. defaultdict: Grouping data without KeyError headaches
# 3. namedtuple: Structured data that's more readable than regular tuples
# They make common data tasks much easier and less error-prone!

# -----------------------------------------------------------------------------
# 🧮 Algorithms and Problem Solving
# -----------------------------------------------------------------------------

# 🤔 What are algorithms and why should I care?
# An algorithm is just a step-by-step recipe for solving a problem efficiently.
# Just like you wouldn't randomly throw ingredients in a pot, you need systematic
# approaches to handle data efficiently, especially when datasets get HUGE!

# 🎯 Why algorithms matter in data science:
# - Wrong algorithm: Takes hours to process data
# - Right algorithm: Takes seconds to process the same data
# - Good algorithms scale well (work with 1000 or 1,000,000 records)
# - Understanding algorithms makes you a better problem solver

print("\n🧮 ALGORITHMS AND PROBLEM SOLVING")
print("-" * 33)

# === SORTING: Organizing Data by Criteria ===
print("🔄 Sorting algorithms:")

# 📊 Sample data: customer records for analysis
# In real life, this might come from a database, CSV file, or API
customers = [
    {"name": "Alice", "age": 28, "purchases": 15, "total_spent": 1500},
    {"name": "Bob", "age": 35, "purchases": 8, "total_spent": 800},
    {"name": "Charlie", "age": 22, "purchases": 25, "total_spent": 2200},
    {"name": "Diana", "age": 31, "purchases": 12, "total_spent": 1100},
]

# 🎯 The magic of key functions in sorting:
# sorted() can sort by ANY criteria using a "key" function
# lambda x: x["age"] means "for each customer x, use x's age as the sorting key"

by_age = sorted(customers, key=lambda x: x["age"])  # Sort youngest to oldest
by_spending = sorted(
    customers, key=lambda x: x["total_spent"], reverse=True
)  # Highest spenders first
by_frequency = sorted(
    customers, key=lambda x: x["purchases"], reverse=True
)  # Most frequent buyers first

print("   Sorted by age:")
for customer in by_age:
    print(f"     {customer['name']} ({customer['age']} years old)")

print("   Top spenders:")
for customer in by_spending[:2]:  # [:2] means "first 2 items"
    print(f"     {customer['name']}: ${customer['total_spent']:,}")

# === BINARY SEARCH: Efficient Searching ===
print("\n🔍 Binary Search - Efficient searching:")

# 🤔 Why is binary search important?
# Linear search: Check every item one by one (like looking through every page of a book)
# Binary search: Start in middle, eliminate half the options each step (like using index!)
# For 1 million items: Linear = 1 million checks, Binary = only 20 checks! 🤯


def binary_search(sorted_list: List[int], target: int) -> int:
    """
    🎯 Find target in sorted list using binary search

    🚀 Why is this so much faster?
    Time complexity: O(log n) instead of O(n)
    - O(n) means time grows linearly with data size
    - O(log n) means time grows much slower (logarithmically)

    For 1,000,000 items:
    - Linear search: up to 1,000,000 checks
    - Binary search: up to 20 checks! 🤯
    """

    left, right = 0, len(sorted_list) - 1  # Start with full range
    steps = 0

    # 🔄 Keep searching until we find it or run out of places to look
    while left <= right:
        steps += 1

        # 📍 Find the middle position
        mid = (left + right) // 2  # // means integer division (no decimal)
        mid_value = sorted_list[mid]

        print(f"     Step {steps}: checking position {mid} (value {mid_value})")

        if mid_value == target:
            # 🎉 Found it! Return the position
            print(f"     ✅ Found {target} at position {mid} in {steps} steps!")
            return mid
        elif mid_value < target:
            # 🔍 Target is bigger, search the right half
            left = mid + 1
        else:
            # 🔍 Target is smaller, search the left half
            right = mid - 1

    # 😞 Not found after checking all possibilities
    print(f"     ❌ {target} not found after {steps} steps")
    return -1  # -1 conventionally means "not found"


# 🧪 Test binary search efficiency
sorted_numbers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
print(f"   Searching for 15 in: {sorted_numbers}")
binary_search(sorted_numbers, 15)

# === PRACTICAL ALGORITHM: Duplicate Detection ===
print("\n🔍 Practical: Finding duplicates in data:")

# 🎯 Why is duplicate detection important in data science?
# - Data quality: Duplicate records skew analysis
# - Customer management: Don't send 3 emails to same person!
# - Inventory: Don't double-count products
# - Survey analysis: One response per person


def analyze_duplicates(data: List[Any]) -> Dict[str, Any]:
    """
    🧹 Analyze data for duplicate entries - essential for data quality!

    🚀 Why use a set for duplicate detection?
    - Checking if item exists in a set: O(1) - instant!
    - Checking if item exists in a list: O(n) - check every item
    For big data, this difference is HUGE!
    """

    seen = set()  # Keep track of items we've already encountered
    duplicates = []  # Store items that appear multiple times

    # 🔄 Check each item in our data
    for item in data:
        if item in seen:
            # 📝 We've seen this before - it's a duplicate!
            duplicates.append(item)
        else:
            # ✅ First time seeing this item
            seen.add(item)

    # 🧮 Calculate useful statistics
    unique_duplicates = list(
        set(duplicates)
    )  # Remove duplicates from duplicates list 😄

    return {
        "original_count": len(data),  # How many items total?
        "unique_count": len(seen),  # How many unique items?
        "duplicate_items": unique_duplicates,  # Which items are duplicated?
        "duplicate_count": len(duplicates),  # How many duplicate occurrences?
        "is_clean": len(duplicates) == 0,  # Is the data clean (no duplicates)?
    }


# 🧪 Test with realistic messy customer data
customer_ids = [101, 102, 103, 102, 104, 105, 103, 106, 101, 107]
duplicate_analysis = analyze_duplicates(customer_ids)

print("   Customer ID analysis:")
print(f"     Original data: {customer_ids}")
print(f"     Total records: {duplicate_analysis['original_count']}")
print(f"     Unique customers: {duplicate_analysis['unique_count']}")
print(f"     Duplicate IDs: {duplicate_analysis['duplicate_items']}")
print(f"     Data is clean: {duplicate_analysis['is_clean']}")

# 🌟 Key algorithm insights:
# 1. Choose the right data structure (set vs list) for better performance
# 2. Understand time complexity (O(1) vs O(n) vs O(log n))
# 3. Algorithms help you process data efficiently, especially at scale
# 4. Good algorithms make the difference between seconds and hours of processing time!

# -----------------------------------------------------------------------------
# 🎯 Putting It All Together
# -----------------------------------------------------------------------------

# 🎯 This is where everything clicks!
# Real-world data science projects combine ALL the concepts we've learned.
# Let's build a complete data analysis pipeline that shows how:
# - String processing cleans messy data
# - Data structures organize information efficiently
# - Control flow handles different scenarios
# - Error handling prevents crashes
# - Algorithms process data efficiently

print("\n🎯 PUTTING IT ALL TOGETHER")
print("-" * 28)


def analyze_customer_data(customers_raw: List[str]) -> Dict[str, Any]:
    """
    📊 Complete data analysis pipeline combining our key concepts:

    🧠 This function demonstrates:
    - String processing (data cleaning)
    - Data structures (organizing information)
    - Control flow (handling different scenarios)
    - Error handling (robust processing)
    - Algorithms (efficient analysis)

    🎯 Input: Raw, messy customer data as strings
    🎯 Output: Clean analysis with insights and error reporting

    This is what real data science looks like!
    """

    # 📦 Initialize our data containers
    customers = []  # List to store clean, validated customer records
    errors = []  # List to track any problems we encounter

    # 🔄 Process each raw customer record (this is data cleaning in action!)
    for i, line in enumerate(
        customers_raw, 1
    ):  # enumerate with 1 gives us line numbers
        try:  # 🛡️ Wrap everything in error handling
            # 🧹 Clean the input data
            line = line.strip()  # Remove extra whitespace
            if not line:  # Skip empty lines
                continue

            # 📝 Parse customer info (expecting: name,age,email,purchases)
            parts = [
                part.strip() for part in line.split(",")
            ]  # Split by commas and clean each part

            # ✅ Validate data structure
            if len(parts) != 4:
                errors.append(f"Line {i}: Wrong number of fields")
                continue  # Skip this record and move to next

            name, age_str, email, purchases_str = parts  # Unpack into variables

            # 🔄 Validate and convert data types (this can fail!)
            age = int(age_str)  # Convert string to integer
            purchases = int(purchases_str)  # Convert string to integer

            # 🧠 Business logic validation (domain-specific rules)
            if age < 0 or age > 120:  # Reasonable age range
                errors.append(f"Line {i}: Invalid age {age}")
                continue

            if purchases < 0:  # Can't have negative purchases
                errors.append(f"Line {i}: Invalid purchases {purchases}")
                continue

            # ✅ Store clean, validated data
            customers.append(
                {
                    "name": name.title(),  # 🎨 Standardize capitalization
                    "age": age,
                    "email": email.lower(),  # 📧 Standardize email format
                    "purchases": purchases,
                }
            )

        # 🛡️ Handle specific error types gracefully
        except ValueError as e:
            # This happens when int() fails (like int("abc"))
            errors.append(f"Line {i}: {e}")
        except Exception as e:
            # Catch any other unexpected errors
            errors.append(f"Line {i}: Unexpected error - {e}")

    # 🚨 Handle edge case: what if no valid data was found?
    if not customers:
        return {"error": "No valid customer data found", "errors": errors}

    # 📊 Analyze the clean data (this is where insights come from!)
    total_customers = len(customers)
    total_purchases = sum(
        c["purchases"] for c in customers
    )  # List comprehension for efficiency
    avg_purchases = total_purchases / total_customers

    # 🏷️ Categorize customers by age groups (business intelligence!)
    # defaultdict automatically creates categories as we encounter them
    age_groups = defaultdict(int)  # int() creates 0 for new keys

    for customer in customers:
        # 🎯 Business logic: categorize by age ranges
        if customer["age"] < 25:
            age_groups["Young (18-24)"] += 1
        elif customer["age"] < 35:
            age_groups["Adult (25-34)"] += 1
        elif customer["age"] < 50:
            age_groups["Middle-aged (35-49)"] += 1
        else:
            age_groups["Senior (50+)"] += 1

    # 🏆 Find top customers by purchase activity (sorting algorithm!)
    top_customers = sorted(customers, key=lambda x: x["purchases"], reverse=True)[:3]

    # 📦 Return comprehensive analysis results
    return {
        "total_customers": total_customers,
        "total_purchases": total_purchases,
        "average_purchases": round(avg_purchases, 1),  # Round for readability
        "age_distribution": dict(age_groups),  # Convert defaultdict to regular dict
        "top_customers": [
            (c["name"], c["purchases"]) for c in top_customers
        ],  # Extract key info
        "errors": errors,
        "data_quality": f"{len(errors)} errors in {len(customers_raw)} records",
    }


# 🧪 Test our complete analysis with realistic messy data
# This simulates what you'd actually get from real-world sources!
raw_customer_data = [
    "Alice Johnson, 28, alice@email.com, 15",  # ✅ Good data
    "Bob Smith, 35, bob@email.com, 8",  # ✅ Good data
    "Charlie Brown, invalid_age, charlie@email.com, 25",  # ❌ Error: invalid age
    "Diana Wilson, 31, diana@email.com, 12",  # ✅ Good data
    "Eve Davis, 45, eve@email.com, 30",  # ✅ Good data
    ", 25, incomplete@email.com, 5",  # ❌ Error: missing name
    "Frank Miller, 29, frank@email.com, 18",  # ✅ Good data
]

print("📈 Complete Customer Analysis:")
results = analyze_customer_data(raw_customer_data)

# 📊 Display results in a readable format
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
# -----------------------------------------------------------------------------
🎯 COMPREHENSIVE KEY TAKEAWAYS
# -----------------------------------------------------------------------------

✅ Variables & Types
   🤔 What are they? Named containers for storing different kinds of information
   🎯 Why important? Make code readable and reusable instead of magic numbers
   💡 Key insight: Python figures out types automatically (dynamic typing)
   🔧 Core types: str (text), int (whole numbers), float (decimals), bool (True/False)
   🛠️ Pro tip: Use type() to check what kind of data you're working with

✅ Strings & Text Processing
   🤔 What's the big deal? Real-world data is messy text that needs cleaning!
   🎯 Why essential? Customer names, addresses, categories all need standardization
   💡 Key methods: .strip() (remove spaces), .lower()/.title() (fix capitalization), .split() (break into pieces)
   🛠️ Pro tip: f-strings are the modern way to format text: f"Hello {name}!"

✅ Conditional Logic (if/elif/else)
   🤔 What do they do? Let programs make decisions based on data conditions
   🎯 Why crucial? Data validation, business rules, filtering - decision-making everywhere!
   💡 Key pattern: if (first condition) elif (second condition) else (everything else)
   🛠️ Pro tip: Use elif instead of multiple ifs for mutually exclusive conditions

✅ Loops & Iteration
   🤔 What problem do they solve? Avoid copy-pasting code 1000 times!
   🎯 Why powerful? Process any amount of data with the same code
   💡 Key tools: for loops (process collections), enumerate() (get position + item), zip() (combine lists)
   🛠️ Pro tip: List comprehensions are Python's superpower for data transformation

✅ Functions
   🤔 What are they? Reusable mini-programs that take input and return output
   🎯 Why essential? Organization, testing, reusability, teamwork
   💡 Key benefits: Write once, use many times; easy to test individual pieces
   🛠️ Pro tip: Use type hints to document what your functions expect and return

✅ Data Structures
   🤔 Why different containers? Different data needs different organization!
   🎯 Choose based on needs:
      - Lists [1,2,3]: Ordered, changeable - great for sequences
      - Dictionaries {"key": "value"}: Fast lookups by key - like phone books
      - Sets {1,2,3}: Unique items only - automatic duplicate removal
      - Tuples (1,2,3): Immutable - good for coordinates, records
   💡 Key insight: Right structure makes code faster and more readable

✅ Advanced Collections
   🤔 Why not just lists and dicts? Specialized tools for common patterns!
   🎯 Power tools:
      - Counter: Automatic frequency counting
      - defaultdict: No more KeyError headaches when grouping data
      - namedtuple: Structured data with named fields (better than unnamed tuples)
   💡 Key benefit: Make common data tasks easier and less error-prone

✅ Error Handling
   🤔 Why not just write perfect code? Real world is messy - things WILL go wrong!
   🎯 What could fail? Missing files, bad data, network issues, user mistakes
   💡 Key pattern: try (attempt something) except (handle problems gracefully)
   🛠️ Pro tip: Return None or default values for recoverable errors, don't crash!

✅ Algorithms & Problem Solving
   🤔 What's an algorithm? Step-by-step recipe for solving problems efficiently
   🎯 Why care? Wrong algorithm = hours of processing, Right algorithm = seconds!
   💡 Key concepts:
      - Time complexity: O(1) instant, O(log n) very fast, O(n) linear
      - Choose right data structure for the job (set for fast lookups)
      - Sorting: Use key functions for custom criteria
   🛠️ Pro tip: Good algorithms scale well from 1000 to 1,000,000 records

🚀 THE BIG PICTURE: Why This All Matters for Data Science
# -----------------------------------------------------------------------------

🎯 Every data science project involves:
   1. 🧹 CLEANING messy data (strings, conditionals, error handling)
   2. 📊 ORGANIZING information efficiently (data structures, algorithms)
   3. 🔄 PROCESSING large datasets (loops, functions, comprehensions)
   4. 🛡️ HANDLING problems gracefully (error handling, validation)
   5. 🧠 ANALYZING patterns (algorithms, sorting, grouping)

🌟 These fundamentals are the building blocks for:
   - pandas (data manipulation)
   - NumPy (numerical computing)
   - scikit-learn (machine learning)
   - matplotlib (visualization)
   - Every other data science tool!

🎉 Congratulations! You now understand the core concepts that power all of data science!

🚀 What's Next?
   - Master these fundamentals through practice
   - Move on to pandas for data manipulation
   - Learn NumPy for numerical computing
   - Start building real projects!

Remember: Every expert was once a beginner. You've got this! 💪
"""
