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

# Stage 4: Forward Propagation  (Classification)

We adapt the forward propagation from Part 1. The structural change is in the **output layer**.

## Key Changes:
1.  **Hidden Layers ($1, 2, 3$):** Continue to use **ReLU** activation.
2.  **Output Layer ($4$):** Uses **Softmax** instead of ReLU. This converts the raw logits ($Z^{[4]}$) into a probability distribution over the 10 classes.

## Formulas:

**Hidden Layers ($l=1,2,3$):**
$$Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = \text{ReLU}(Z^{[l]})$$

**Output Layer ($l=4$):**
$$Z^{[4]} = W^{[4]} \cdot A^{[3]} + b^{[4]}$$
$$A^{[4]} = \text{Softmax}(Z^{[4]})$$

---

# Stage 5: Backpropagation for Classification

In Part 2, the Backpropagation logic changes specifically at the **Output Layer** due to the combination of **Softmax** activation and **Cross-Entropy** loss.

## 1. Gradient at Output Layer (Layer 4)
Unlike the Regression part (MSE), we do not need to compute $dA^{[4]}$ explicitly. The derivative of the Cross-Entropy loss with respect to the Softmax input ($Z^{[4]}$) simplifies to an elegant subtraction:

$$
dZ^{[4]} = \frac{1}{m} (A^{[4]} - Y)
$$

Where:
- $A^{[4]}$: Predicted probabilities (Output of Softmax).
- $Y$: One-Hot encoded true labels.
- $m$: Number of training examples.

*Note: This simplifies the calculation significantly compared to computing the Jacobian of Softmax manually.*

## 2. Propagating Error (Layers 3 $\rightarrow$ 1)
For the hidden layers, the logic remains identical to Part 1 (Regression), as they still use **ReLU** activation.

**For each hidden layer $l$ (where $l=3, 2, 1$):**

1.  **Error from next layer:**
    $$dA^{[l]} = W^{[l+1]T} \cdot dZ^{[l+1]}$$

2.  **Apply ReLU Derivative:**
    $$dZ^{[l]} = dA^{[l]} * g'(Z^{[l]})$$
    *(Where $g'(z)$ is 1 if $z > 0$, else 0)*

3.  **Gradients for Weights and Biases:**
    $$dW^{[l]} = dZ^{[l]} \cdot A^{[l-1]T} + \frac{\lambda}{m} W^{[l]}$$
    $$db^{[l]} = \sum_{i=1}^{m} dZ^{[l](i)}$$

## Summary of Architecture Flow
- **Forward:** Input $\xrightarrow{ReLU}$ Hidden $\xrightarrow{ReLU}$ Hidden $\xrightarrow{ReLU}$ Output $\xrightarrow{Softmax}$ Probabilities
- **Backward:** (Probabilities - Labels) $\xrightarrow{Backprop}$ Update Parameters


# Stage6 : Training Pipeline (Classification)

This stage integrates all previous components (Softmax, Cross-Entropy, Backpropagation) into a unified training loop using Stochastic Gradient Descent (SGD).

##  Objective
To train the neural network on the MNIST dataset by iteratively updating parameters to minimize the Cross-Entropy loss and maximize classification accuracy.

##  The Training Loop Logic
The training process differs slightly from Regression (Part 1) by including **Accuracy** as a performance metric and using the Classification-specific functions.

### Process Flow (Per Iteration):

1.  **Forward Pass (`feed_forward_clf`):**
    *   Compute activations for all layers.
    *   Apply **Softmax** at the output to get probabilities $\hat{Y}$.
    *   *Cache* intermediate values ($Z, A$) for backpropagation.

2.  **Loss Computation (`cross_entropy_loss`):**
    *   Calculate the discrepancy between predictions $\hat{Y}$ and true One-Hot labels $Y$.
    *   $$J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} Y_k^{(i)} \log(\hat{Y}_k^{(i)})$$

3.  **Backpropagation (`Backpropagation_clf`):**
    *   Compute gradients ($\nabla W, \nabla b$) starting from the simplified Softmax error: $dZ^{[4]} = \frac{1}{m}(A^{[4]} - Y)$.
    *   Apply L2 Regularization derivative to weights inside this step.

4.  **Parameter Update (SGD):**
    *   Update weights and biases using the learning rate $\eta$:
    *   $$\theta \leftarrow \theta - \eta \cdot \nabla \theta$$

5.  **Monitoring (New Feature):**
    *   **Loss:** Track Cross-Entropy loss history.
    *   **Accuracy:** Periodically convert probabilities to class labels using `argmax` and compare with ground truth to report model performance (e.g., "92% Accuracy").

## Key Formulas

### Parameter Update Rule
For every weight $W$ and bias $b$ in layer $l$:
$$W^{[l]} = W^{[l]} - \eta \cdot dW^{[l]}$$
$$b^{[l]} = b^{[l]} - \eta \cdot db^{[l]}$$

### Accuracy Calculation
$$\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}(\text{argmax}(\hat{y}^{(i)}) == \text{argmax}(y^{(i)}))$$

##  Checklist
- [x] Forward pass returns probability distribution (sums to 1).
- [x] Loss decreases over iterations.
- [x] Accuracy increases over iterations.
- [x] Parameters are updated in-place (or correctly reassigned).

