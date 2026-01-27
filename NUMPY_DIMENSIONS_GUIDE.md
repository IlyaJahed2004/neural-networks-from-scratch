# NumPy Masterclass: Shapes, Dimensions & Broadcasting

This document serves as a comprehensive reference guide for handling array shapes, dimensions, and broadcasting rules in NumPy. Understanding these concepts is **critical** for implementing Neural Networks from scratch, as 90% of bugs usually stem from shape mismatches.

---

## 1. The "Rank-1" Array Trap `(n,)`

In linear algebra, we typically work with **Column Vectors** ($n \times 1$) or **Row Vectors** ($1 \times n$).
However, NumPy introduces a third type: the **Rank-1 Array** or "Shape-less Vector".

### The Problem
A Rank-1 array has a shape of `(n,)`. It implies directionless data.
```python
import numpy as np

a = np.array([1, 2, 3])
print(a.shape)  # Output: (3,) -> One dimension only!

**Why is it dangerous?**
1.  **Transpose does nothing:** `a.T` is still `(3,)`. It does not become a column vector.
2.  **Ambiguous Broadcasting:** When interacting with matrices, NumPy has to "guess" how to align it, which can lead to silent logical errors.

### The Solution: Enforce Dimensions
Always be explicit about your dimensions.

| Type | Shape | Code Example |
| :--- | :--- | :--- |
| **Rank-1 (Avoid)** | `(n,)` | `np.array([1, 2, 3])` |
| **Row Vector** | `(1, n)` | `np.array([[1, 2, 3]])` or `arr.reshape(1, -1)` |
| **Column Vector** | `(n, 1)` | `np.array([[1], [2], [3]])` or `arr.reshape(-1, 1)` |

---

## 2. Reshaping & NewAxis

If you accidentally create a Rank-1 array, you can fix it using `reshape` or `np.newaxis`.

python
x = np.random.randn(5) # Shape: (5,) -> Bad for matrix math

# Method 1: Reshape
x_col = x.reshape(-1, 1) # Shape: (5, 1) -> Column Vector
x_row = x.reshape(1, -1) # Shape: (1, 5) -> Row Vector

# Method 2: NewAxis (Cleaner syntax)
x_col_2 = x[:, np.newaxis] # Adds a new axis at the end -> (5, 1)
x_row_2 = x[np.newaxis, :] # Adds a new axis at the start -> (1, 5)

---

## 3. Broadcasting: The Golden Rules

Broadcasting allows NumPy to perform operations on arrays with different shapes. It "stretches" the smaller array to match the larger one without actually copying data.

**The Rule:** NumPy compares shapes starting from the **last dimension (rightmost)** and moving backward. Two dimensions are compatible if:
1.  They are **equal**, OR
2.  One of them is **1**.

### Example 1: Scalar Broadcasting (Trivial)
python
A = np.ones((2, 3))  # Shape: (2, 3)
b = 5                # Shape: (1,) roughly
C = A + b            # b is stretched to (2, 3). Result is all 6s.

### Example 2: Column Vector Broadcasting (Correct Usage)
We want to add a bias value to each row.

python
matrix = np.ones((3, 4))  # Shape: (3, 4)
bias   = np.arange(3).reshape(3, ** | `(1, n)` | `np.array([[1, 2, 3]])` or `arr.reshape(1, -1)` |
| **Column Vector** | `(n, 1)` | `np.array([[1], [2], [3]])` or `arr.reshape(-1, 1)` |

---

## 2. Reshaping & NewAxis

If you accidentally create a Rank-1 array, you can fix it using `reshape` or `np.newaxis`.

```python
x = np.random.randn(5) # Shape: (5,) -> Bad for matrix math

# Method 1: Reshape
x_col = x.reshape(-1, 1) # Shape: (5, 1) -> Column Vector
x_row = x.reshape(1, -1) # Shape: (1, 5) -> Row Vector

# Method 2: NewAxis (Cleaner syntax)
x_col_2 = x[:, np.newaxis] # Adds a new axis at the end -> (5, 1)
x_row_2 = x[np.newaxis, :] # Adds a new axis at the start -> (1, 5)

---

## 3. Broadcasting: The Golden Rules

Broadcasting allows NumPy to perform operations on arrays with different shapes. It "stretches" the smaller array to match the larger one without actually copying data.

**The Rule:** NumPy compares shapes starting from the **last dimension (rightmost)** and moving backward. Two dimensions are compatible if:
1.  They are **equal**, OR
2.  One of them is **1**.

### Example 1: Scalar Broadcasting (Trivial)
python
A = np.ones((2, 3))  # Shape: (2, 3)
b = 5                # Shape: (1,) roughly
C = A + b            # b is stretched to (2, 3). Result is all 6s.

### Example 2: Column Vector Broadcasting (Correct Usage)
We want to add a bias value to each row.

python
matrix = np.ones((3, 4))  # Shape: (3, 4)
bias   = np.arange(3).reshape(3, 1) # Shape: (3, 1)

# Comparison:
# Matrix: (3, 4)
# Bias:   (3, 1)
#          ^  ^
#          |  Matches because one is 1 (Bias stretches across columns)
#          Matches (3 == 3)

result = matrix + bias # Shape: (3, 4)

### Example 3: The "Dimension Mismatch" Error
```python
matrix = np.ones((3, 4)) # Shape: (3, 4)
vec    = np.arange(3)    # Shape: (3,)

# Comparison:
# Matrix: (3, 4)
# Vec:       (3,)
#             ^
#             Mismatch! 4 != 3.
# output = matrix + vec -> ValueError: operands could not be broadcast together

---

## 4. The Power of `keepdims=True`

When performing reduction operations like `sum`, `mean`, or `max`, NumPy removes the dimension you operated on by default. This creates Rank-1 arrays and breaks broadcasting.

### The "Collapse" Problem
```python
A = np.random.randn(10, 5) # (10 classes, 5 examples)

sum_A = np.sum(A, axis=0) # Sum over rows (collapsing vertical axis)
print(sum_A.shape) 
# Output: (5,) -> Rank-1 Array. The row dimension is GONE.

### The Fix
```python
sum_A_keep = np.sum(A, axis=0, keepdims=True)
print(sum_A_keep.shape) 
# Output: (1, 5) -> Row dimension is kept as 1.

### Why use `keepdims`?
It allows for immediate broadcasting in subsequent steps, like normalization (Softmax).

```python
# Softmax logic requiring division
# A: (10, m)
# sum: must be broadcastable to (10, m). 

# Correct:
total = np.sum(A, axis=0, keepdims=True) # (1, m)
probs = A / total # (10, m) / (1, m) -> Works!

# Incorrect:
total_bad = np.sum(A, axis=0) # (m,)
probs = A / total_bad 
# (10, m) / (m,) -> Compares m vs m (ok), then 10 vs nothing? 
# This might work unpredictably or fail depending on m.

---

## 5. Matrix Multiplication (`@`) vs. Element-wise (`*`)

This is a frequent source of confusion.

### `*` Operator: Element-wise Multiplication
Requires shapes to be identical or broadcastable.
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 2], [2, 2]])

C = A * B 
# [[2, 4], [6, 8]] -> Multiplies corresponding elements.

### `@` Operator (or `np.dot`): Matrix Multiplication
Follows Linear Algebra rules: $(n, k) \times (k, m) = (n, m)$.
```python
# A: (2, 2)
# B: (2, 2)
D = A @ B 
# Row 1 of A dot Column 1 of B...

---

## Summary Cheat Sheet

1.  **Avoid `(n,)`**: If you see this shape, immediately ask yourself: "Is this a row or a column?" and reshape it.
2.  **Debug with `.shape`**: Before fixing a bug, print the shape of every variable involved in the operation.
3.  **Use `keepdims=True`**: Whenever you `sum` or `max` inside a neural network layer, use this to preserve alignment.
4.  **Right-to-Left**: Check broadcasting compatibility starting from the last dimension.

