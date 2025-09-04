

```
╔───────────────────────────────────────────────────────────────────────────────────────────────────────────╗
│                                                                                                           │
│ ██████╗ ██╗   ██╗████████╗██╗  ██╗ ██████╗ ███╗   ██╗                                                     │
│ ██╔══██╗╚██╗ ██╔╝╚══██╔══╝██║  ██║██╔═══██╗████╗  ██║                                                     │
│ ██████╔╝ ╚████╔╝    ██║   ███████║██║   ██║██╔██╗ ██║                                                     │
│ ██╔═══╝   ╚██╔╝     ██║   ██╔══██║██║   ██║██║╚██╗██║                                                     │
│ ██║        ██║      ██║   ██║  ██║╚██████╔╝██║ ╚████║                                                     │
│ ╚═╝        ╚═╝      ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝                                                     │
│                                                                                                           │
│ ███████╗██╗   ██╗███╗   ██╗██████╗  █████╗ ███╗   ███╗███████╗███╗   ██╗████████╗ █████╗ ██╗     ███████╗ │
│ ██╔════╝██║   ██║████╗  ██║██╔══██╗██╔══██╗████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██║     ██╔════╝ │
│ █████╗  ██║   ██║██╔██╗ ██║██║  ██║███████║██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ███████║██║     ███████╗ │
│ ██╔══╝  ██║   ██║██║╚██╗██║██║  ██║██╔══██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ██╔══██║██║     ╚════██║ │
│ ██║     ╚██████╔╝██║ ╚████║██████╔╝██║  ██║██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ██║  ██║███████╗███████║ │
│ ╚═╝      ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝ │
│                                                                                                           │
╚───────────────────────────────────────────────────────────────────────────────────────────────────────────╝
```


# 🐍 1 — Python Fundamentals

**⏱️ Time**: 2-3 hours | **🎯 Difficulty**: 🟢 Beginner | **📁 Files**: `.py` + `.ipynb`

## 📱 30-Second Summary
**Learn Python for data science in 3 hours.** Master variables, strings, functions, and data structures through real examples like cleaning customer data and handling errors. Skip "hello world" — go straight to practical skills you'll use daily in data work.

**🎯 You'll build**: Text cleaning functions, error-handling data processors, and reusable code patterns.

---

Okay, here's the deal. I spent way too many hours figuring out Python the hard way — bouncing between Stack Overflow, half-baked tutorials, and docs that assumed I was already a dev.
This is what I *wish* I had when I started.

---

## 🤷‍♂️ What is this?

This is Python boiled down to what actually matters for **anyone doing data work**.
No fluff. No “hello world” exercises. Just the stuff you’ll use *every single day*.

**What you’ll learn:**
- How Python actually thinks about data (spoiler: it’s smarter than you think)
- String methods that’ll save your life when dealing with messy CSVs
- The logic you need for filtering, looping, and wrangling
- Functions — because copy-pasting the same code is a bad time
- Data containers that don’t suck: `lists`, `dicts`, `sets`, `tuples`
- How to stop your script from crashing like a toddler in a toy store
- Some clever built-in tricks that’ll make you feel like a wizard 🧙

---

## 🎯 Why should you care?

Because every cool data science or ML thing — from `pandas` to `scikit-learn` — *builds on this*.
I learned that the hard way after wasting hours debugging code that broke because I misunderstood Python basics.

After going through this, you’ll actually *get* what your code is doing — not just cross your fingers and hope it runs.
Bonus: you'll stop making those silly mistakes that every beginner makes (and secretly Googles 👀).

---

## 🏃‍♂️ How to run it

```bash
python 1_python_fundamentals.py
```


That’s it. No setup. Just run it top to bottom.
Everything’s explained with real examples, not textbook nonsense.

---

## 🗺️ Where this fits

You're here → [`brainbytes-ds/1_python_fundamentals/`](../1_python_fundamentals/)
Continue the journey:

- 📦 [`2_data_manipulation/`](../2_data_manipulation/) — wrangle real datasets
- 📊 [`3_data_visualization/`](../3_data_visualization/) — make sense of the mess
- 🤖 [`4_statistics_ml/`](../4_statistics_ml/) — the fun predictive stuff


---

## 💭 Worth a skim even if you’re not a beginner

Even if you *think* you know Python, there’s probably something here you didn’t know you needed.
I packed in those “OH, so *that’s* how it works” moments — the ones that took me forever to figure out.

Also: the examples are grounded in real use cases — think inventory systems, sales data, and messy user input —
not geometry formulas or calculator apps.

---

## 📚 Quick Reference Cheat Sheet

### 🧵 Essential String Methods
```python
# Clean messy text (use these daily!)
name = "  JOHN DOE  "
clean = name.strip().title()  # "John Doe"
email = name.lower().replace(" ", ".") + "@company.com"

# Split and join
words = "apple,banana,orange".split(",")  # ['apple', 'banana', 'orange']
sentence = " ".join(words)  # "apple banana orange"
```

### 🗂️ Data Structure Quick Picks
```python
# When to use what:
my_list = [1, 2, 3, 2]        # Order matters, allows duplicates
my_set = {1, 2, 3}            # Unique items only, fast lookups
my_dict = {"key": "value"}    # Key-value pairs, fast access
my_tuple = (1, 2, 3)          # Immutable, perfect for coordinates
```

### 🛡️ Error Handling Pattern
```python
# Always use this pattern for data processing
try:
    result = risky_operation(data)
except ValueError as e:
    print(f"Data error: {e}")
    result = default_value
except Exception as e:
    print(f"Unexpected error: {e}")
    result = None
```

### 🎯 Function Template
```python
def clean_data(raw_data, default_value=""):
    """
    Clean and normalize input data.
    Args: raw_data (str), default_value (str)
    Returns: str (cleaned data)
    """
    if not raw_data:
        return default_value
    return raw_data.strip().lower()
```

---

## 🧪 Can You Do This? (Self-Check)

**🟢 Beginner Checkpoints**:
- [ ] Create a function that takes a messy name and returns "First Last" format
- [ ] Use a dictionary to count word frequencies in a sentence
- [ ] Handle a potential division-by-zero error gracefully
- [ ] Clean a list of email addresses (lowercase, no spaces)

**🟡 Intermediate Challenges**:
- [ ] Build a function that validates email format using string methods
- [ ] Create a data processor that handles multiple file formats
- [ ] Use Counter to analyze the most common problems in a dataset
- [ ] Write error-proof code that never crashes on bad input

**Ready for Module 2?** ✅ You should feel confident with strings, functions, and basic error handling!

---

_Made this with a lot of trial-and-error, some help from Claude when I hit a wall, and too much caffeine._
_Hope it saves you some time (and sanity)._ ✌️

---
***Navigation:***<br>
[🏠 Home](../) › **📂 Python Fundamentals** › [📂 Data Manipulation](../2_data_manipulation/) › [📂 Data Visualization](../3_data_visualization/) › [📂 Statistics & ML](../4_statistics_ml/)
