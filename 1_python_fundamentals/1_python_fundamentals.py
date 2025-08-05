"""
🐍 Python Fundamentals Reference & Cheatsheet
Comprehensive executable examples for syntax, data structures, and algorithms
"""

# =============================================================================
# 📦 MODULE IMPORTS — Core Python & Typing Utilities
# =============================================================================

from typing import List, Optional, Dict, Any
from collections import namedtuple, defaultdict, Counter, deque


# =============================================================================
# 📝 SYNTAX & BASIC PROGRAMMING
# =============================================================================

# -- Variables & Type Checking --
name = "Alice"
age = 30
salary = 75000.50
is_employed = True

print(f"📊 {name} is {age} years old, earns ${salary:,.2f}, employed: {is_employed}")
print(f"🔍 Types: {type(name).__name__}, {type(age).__name__}, {type(salary).__name__}")

# -- String Operations (JS dev will appreciate f-strings) --
first_name, last_name = "John", "Doe"
full_name = f"{first_name} {last_name}"
email = f"{first_name.lower()}.{last_name.lower()}@company.com"
print(f"📧 Generated email: {email}")


# -- Conditional Logic with Walrus Operator (Python 3.8+) --
def check_grade(score):
    """🎯 Grade checker with modern Python syntax"""
    if (grade := score) >= 90:
        return f"A ({grade})"
    elif grade >= 80:
        return f"B ({grade})"
    elif grade >= 70:
        return f"C ({grade})"
    else:
        return f"F ({grade})"


scores = [95, 87, 72, 65]
for score in scores:
    print(f"📊 Score {score}: {check_grade(score)}")

# -- Loops with Enumerate & Zip (common patterns) --
products = ["laptop", "mouse", "keyboard"]
prices = [999, 25, 75]

print("\n💰 Product Catalog:")
for idx, (product, price) in enumerate(zip(products, prices), 1):
    print(f"{idx}. {product.capitalize()}: ${price}")

# -- Functions with Type Hints & Default Values --
# from typing import List, Optional, Dict, Any


def calculate_tax(
    amount: float, rate: float = 0.08, currency: str = "USD"
) -> Dict[str, Any]:
    """💸 Calculate tax with flexible parameters"""
    tax = amount * rate
    total = amount + tax
    return {
        "original": f"{amount:.2f} {currency}",
        "tax": f"{tax:.2f} {currency}",
        "total": f"{total:.2f} {currency}",
        "rate_percent": f"{rate * 100}%",
    }


# Test the function
tax_result = calculate_tax(100.0, 0.1)
print(f"\n🧾 Tax Calculation: {tax_result}")


# -- Error Handling Patterns --
def safe_divide(a: float, b: float) -> Optional[float]:
    """🛡️ Safe division with error handling"""
    try:
        result = a / b
        print(f"✅ {a} ÷ {b} = {result}")
        return result
    except ZeroDivisionError:
        print(f"❌ Cannot divide {a} by zero!")
        return None
    except TypeError as e:
        print(f"❌ Type error: {e}")
        return None


# Test error handling
safe_divide(10, 2)
safe_divide(10, 0)
safe_divide("10", 2)

# =============================================================================
# 📚 INTERMEDIATE & ADVANCED DATA STRUCTURES
# =============================================================================

# -- List Comprehensions & Filtering --
numbers = list(range(1, 21))
print(f"\n🔢 Original numbers: {numbers}")

# Multiple list comprehension patterns
evens = [n for n in numbers if n % 2 == 0]
squares = [n**2 for n in numbers if n <= 10]
even_squares = [n**2 for n in numbers if n % 2 == 0 and n <= 10]

print(f"🔵 Even numbers: {evens}")
print(f"🟡 Squares (≤10): {squares}")
print(f"🟢 Even squares (≤10): {even_squares}")

# -- Dictionary Comprehensions & Transformations --
sales_data = {"Jan": 1000, "Feb": 1200, "Mar": 800, "Apr": 1500}
print(f"\n💼 Sales Data: {sales_data}")

# Dictionary transformations
sales_k = {month: f"${amount / 1000:.1f}K" for month, amount in sales_data.items()}
high_months = {month: amount for month, amount in sales_data.items() if amount > 1000}
quarter_total = sum(sales_data.values())

print(f"📊 Sales in K: {sales_k}")
print(f"🔥 High performing months: {high_months}")
print(f"📈 Q1 Total: ${quarter_total:,}")

# -- Sets for Data Operations --
team_a = {"Alice", "Bob", "Charlie", "David"}
team_b = {"Charlie", "David", "Eve", "Frank"}

print(f"\n👥 Team A: {team_a}")
print(f"👥 Team B: {team_b}")
print(f"🤝 Both teams: {team_a & team_b}")  # Intersection
print(f"🆚 Only A: {team_a - team_b}")  # Difference
print(f"🔄 All members: {team_a | team_b}")  # Union

# -- Tuples for Immutable Data --
# Named tuples for structured data
# from collections import namedtuple

Employee = namedtuple("Employee", ["name", "department", "salary"])

employees = [
    Employee("Alice", "Engineering", 90000),
    Employee("Bob", "Marketing", 70000),
    Employee("Charlie", "Engineering", 95000),
]

print("\n👨‍💼 Employee Records:")
for emp in employees:
    print(f"  {emp.name} ({emp.department}): ${emp.salary:,}")

# Group by department using dictionaries
# from collections import defaultdict

by_dept = defaultdict(list)
for emp in employees:
    by_dept[emp.department].append(emp)

for dept, emp_list in by_dept.items():
    avg_salary = sum(emp.salary for emp in emp_list) / len(emp_list)
    print(f"📊 {dept}: {len(emp_list)} employees, avg salary: ${avg_salary:,.0f}")

# -- Advanced Collection Operations --
# from collections import Counter, deque

# Counter for frequency analysis
text = "hello world hello python world"
word_freq = Counter(text.split())
print(f"\n📝 Word frequencies: {dict(word_freq)}")
print(f"🔝 Most common: {word_freq.most_common(2)}")

# Deque for efficient operations at both ends
recent_actions = deque(maxlen=5)  # Keep only last 5 actions
for action in ["login", "view_profile", "edit_data", "save", "logout", "login_again"]:
    recent_actions.append(action)
    print(f"🔄 Recent actions: {list(recent_actions)}")

# =============================================================================
# 🧮 ALGORITHM BASICS
# =============================================================================

# -- Sorting Examples --
products_data = [
    {"name": "Laptop", "price": 999, "rating": 4.5},
    {"name": "Mouse", "price": 25, "rating": 4.2},
    {"name": "Keyboard", "price": 75, "rating": 4.7},
    {"name": "Monitor", "price": 300, "rating": 4.4},
]

print(f"\n🛍️ Original products: {[p['name'] for p in products_data]}")

# Multiple sorting criteria
by_price = sorted(products_data, key=lambda x: x["price"])
by_rating = sorted(products_data, key=lambda x: x["rating"], reverse=True)
by_value = sorted(
    products_data, key=lambda x: x["rating"] / (x["price"] / 100), reverse=True
)

# print(f"💰 By price: {[f\"{p['name']} (${p['price']})\" for p in by_price]}")
# print(f"⭐ By rating: {[f\"{p['name']} ({p['rating']})\" for p in by_rating]}")
# print(f"🎯 By value: {[f\"{p['name']} (value score)\" for p in by_value]}")

print(f"💰 By price: {[f'{p["name"]} (${p["price"]})' for p in by_price]}")
print(f"⭐ By rating: {[f'{p["name"]} ({p["rating"]})' for p in by_rating]}")
print(f"🎯 By value: {[f'{p["name"]} (value score)' for p in by_value]}")


# -- Binary Search Implementation --
def binary_search(arr: List[int], target: int) -> int:
    """🔍 Binary search with detailed logging"""
    left, right = 0, len(arr) - 1
    steps = 0

    while left <= right:
        steps += 1
        mid = (left + right) // 2
        mid_val = arr[mid]

        print(f"  Step {steps}: checking index {mid} (value {mid_val})")

        if mid_val == target:
            print(f"  ✅ Found {target} at index {mid} in {steps} steps")
            return mid
        elif mid_val < target:
            left = mid + 1
            print(f"  🔺 Target {target} > {mid_val}, searching right half")
        else:
            right = mid - 1
            print(f"  🔻 Target {target} < {mid_val}, searching left half")

    print(f"  ❌ Target {target} not found after {steps} steps")
    return -1


# Test binary search
sorted_numbers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
print(f"\n🔍 Binary Search in: {sorted_numbers}")
binary_search(sorted_numbers, 7)
binary_search(sorted_numbers, 12)


# -- Recursion Examples --
def fibonacci_memo(n: int, memo: Dict[int, int] = None) -> int:
    """🌀 Fibonacci with memoization for performance"""
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]


# Generate Fibonacci sequence
fib_sequence = [fibonacci_memo(i) for i in range(10)]
print(f"\n🌀 Fibonacci sequence: {fib_sequence}")


# -- Practical Algorithm: Find Duplicates --
def find_duplicates(data: List[Any]) -> Dict[str, Any]:
    """🔄 Find duplicates with statistics"""
    seen = set()
    duplicates = []

    for item in data:
        if item in seen:
            duplicates.append(item)
        else:
            seen.add(item)

    return {
        "duplicates": list(set(duplicates)),
        "duplicate_count": len(duplicates),
        "unique_count": len(seen),
        "total_items": len(data),
    }


# Test duplicate finder
test_data = [1, 2, 3, 2, 4, 5, 3, 6, 1, 7]
dup_result = find_duplicates(test_data)
print(f"\n🔄 Duplicate analysis: {dup_result}")


# -- Quick Sort Implementation --
def quicksort(arr: List[int]) -> List[int]:
    """⚡ Quick sort with list comprehensions"""
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


# Test sorting
unsorted_data = [64, 34, 25, 12, 22, 11, 90]
sorted_data = quicksort(unsorted_data)
print(f"\n⚡ Quick sort: {unsorted_data} → {sorted_data}")

print(
    "\n🎉 Python Fundamentals Complete! Check other files for data manipulation & ML."
)

# %%
