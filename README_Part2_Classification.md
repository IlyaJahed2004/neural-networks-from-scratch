# Part 2: Multiclass Classification (MNIST)

In this second phase of the project, we transition from regression to a multiclass classification task. The goal is to train a Neural Network to recognize handwritten digits (0-9) using the famous MNIST dataset.

While the core "engine" of our MLP remains similar to Part 1, the input pipeline and output interpretation require significant structural changes.

## Stage 1: Data Loading & Preprocessing Pipeline

This stage establishes the foundation for image classification by transforming raw pixel data into a format suitable for a Fully Connected Network.

### Key Preprocessing Steps:

*   **Dataset Acquisition (MNIST):**
    We utilize the MNIST dataset, consisting of 70,000 greyscale images of handwritten digits.
    - **Features (X):** Each image is a $28 \times 28$ grid of pixels (784 total pixels).
    - **Labels (Y):** The target is a digit from 0 to 9.

*   **Flattening (Image to Vector):**
    Since we are using a standard MLP (not a CNN yet), we cannot feed 2D images directly. We "flatten" each $28 \times 28$ image into a single column vector of size $784 \times 1$. This allows the first layer of the network to treat each pixel as an individual input feature.

*   **Pixel Normalization:**
    Raw pixel values range from 0 to 255. We scale these values to the range $[0, 1]$ by dividing by 255.0. This is crucial because:
    - It prevents numerical instability.
    - It ensures gradients don't explode/vanish initially.
    - It aligns with the initialization scale of our weights.

*   **One-Hot Encoding (Target Transformation):**
    Unlike regression where the output is a single scalar (e.g., popularity), classification requires a probability distribution. We convert the integer labels (e.g., "5") into One-Hot vectors (e.g., `[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]`).
    - This corresponds to the 10 neurons in the output layer.
    - It allows us to calculate the error for each class probability separately.

*   **Architecture Alignment (Transposition):**
    Consistent with Part 1, we transpose the matrices to shape $(features, samples)$.
    - Input Matrix Shape: $(784, m)$
    - Output Matrix Shape: $(10, m)$
    where $m$ is the number of examples in the batch.

### Summary
At the end of this stage, the raw image data is fully preprocessed, normalized, and shaped correctly, ready to be fed into the neural network for initialization.

## Stage 2: Softmax Activation & Numerical Stability

In this stage, we implement the **Softmax** activation function, which is the standard output layer for multi-class classification problems.

While ReLU is used for hidden layers to introduce non-linearity, Softmax is required at the output to interpret the raw network scores (logits) as **probabilities**.

### Softmax Formula

For a given input vector $z$ (logits for one sample), the Softmax function is defined as:

$$
\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

Where:
- $K$ is the number of classes (10 for MNIST).
- The output values are in range $(0, 1)$.
- The sum of all output values for a single sample equals $1$.

### Matrix Implementation Logic

Since our data is structured as `(features/classes, samples)`, we perform operations along **axis 0** (vertical):

1.  **Exponentiation:** Calculate $e^Z$ element-wise.
2.  **Summation:** Sum the exponential values down the columns (per sample).
    *   Crucial Note: We use `keepdims=True` to maintain the matrix rank, resulting in a shape of `(1, m)` instead of a rank-1 array `(m,)`.
3.  **Normalization:** Divide the exponentials by the sum.

### Numerical Stability (The "Max Trick")

Computing $e^z$ for large values of $z$ (e.g., $z > 700$) can cause a numerical overflow, returning `inf` or `NaN`. To prevent this, we use the property that Softmax is translation-invariant:

$$
\frac{e^{z_i}}{\sum e^{z_j}} = \frac{e^{z_i - C}}{\sum e^{z_j - C}}
$$

We set $C = \max(Z)$ (column-wise max). By subtracting the maximum value from each column before exponentiation:
- The largest value becomes 0 ($e^0 = 1$).
- All other values become negative ($e^{-x}$ is small but safe).
- **Result:** The mathematical output is identical, but numerically stable.

### Scope of This Stage

`Implemented`:
- Softmax function with `axis=0` summation.
- Numerical stability fix (subtracting max).
- Handling of matrix dimensions for correct broadcasting.

`Not implemented`:
- Cross-Entropy Loss (Stage 3).
- Backpropagation for Softmax (Stage 3).

# Stage3: Cross-Entropy Loss

In Part 2 (Classification), we use **Cross-Entropy Loss**.

## 1. Why change the loss function?
In regression (Part 1), "close enough" was good (e.g., predicting 0.8 when the answer is 1.0 is decent).
In classification, we deal with probabilities. If the image is a **"7"** but our network assigns a probability of $0.001$ to class "7", it is **dead wrong** and should be penalized heavily.

**Cross-Entropy** effectively penalizes confident but wrong predictions.

---

## 2. Mathematical Formulation

$$L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{O} Y_j^{(i)} \log(A_j^{[L](i)})$$

Where:
*   $m$: Number of training examples (batch size).
*   $O$: Number of output classes (10 for MNIST).
*   $Y$: The Ground Truth (One-Hot Encoded). Only one element is 1, the rest are 0.
*   $A^{[L]}$: The predicted probabilities (Output of Softmax).

### How it works intuitively?
Since $Y$ is one-hot (e.g., `[0, 1, 0]`), the inner sum only "cares" about the correct class.
*   If correct class prob ($A$) is 1.0 $\rightarrow$ $\log(1) = 0$ $\rightarrow$ **Loss is 0** (Perfect).
*   If correct class prob ($A$) is 0.1 $\rightarrow$ $\log(0.1) = -2.3$ $\rightarrow$ **Loss is High**.
*   If correct class prob ($A$) is 0.0 $\rightarrow$ $\log(0) = -\infty$ $\rightarrow$ **Disaster!**

---
