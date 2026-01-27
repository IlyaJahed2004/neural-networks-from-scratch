# Neural Networks From Scratch (NumPy)

This repository contains an incremental implementation of a Multi-Layer Perceptron (MLP)
built entirely from scratch using NumPy. The project emphasizes a clear, modular
development process, with each core component of the neural network being implemented
step-by-step and tracked through isolated commits.

##  Stage 1: Data Preprocessing Pipeline

This initial commit focuses on establishing a robust data loading and preprocessing
pipeline for a regression task: predicting song popularity using the Spotify dataset.

### Implemented Components:
*   **Dataset Loading and Cleaning:** Handling of `release_date` to extract `year`,
    and robust conversion of relevant columns to numeric types using `errors='coerce'`.
*   **Feature Selection:** Identification and selection of pertinent audio features
    and the target variable (`popularity`).
*   **Handling Missing Values:** Strategic removal of rows with `NaN` values in critical
    features or the target, ensuring data integrity for model training.
*   **Min-Max Normalization:** Scaling of all input features and the target to a `[0, 1]` range
    using `MinMaxScaler`, crucial for stabilizing neural network training.
*   **Train-Test Split:** Partitioning the dataset into training and testing sets
    to enable objective model evaluation and prevent overfitting.
*   **Input Formatting for Neural Networks:** Transposing the data (`(samples, features)` to `(features, samples)`)
    to align with the standard mathematical conventions for weight-activation matrix multiplications
    in neural network architectures ($Z = W \cdot A + b$).


## Project Philosophy

Each core component of the neural network is implemented and committed in a focused manner,
allowing for a clean Git history that reflects the incremental development of a complex system.
This approach facilitates easier review, debugging, and understanding of the project's evolution.


## Stage 2 — He Initialization

In this stage, we focus **only on parameter initialization** for the neural network.
No forward pass, loss function, or training logic is implemented yet.

Since the network uses **ReLU activations** in all layers, proper initialization is critical
to ensure stable signal propagation and effective learning.

---

### Why Initialization Matters

Poor weight initialization can lead to:

- **Vanishing gradients** (signals shrink across layers)
- **Exploding gradients** (signals grow uncontrollably)
- Dead ReLU neurons
- Extremely slow or unstable training

These issues become more severe as the network depth increases.

---

### He Initialization (Kaiming Initialization)

To address these problems, we use **He Initialization**, which is specifically designed
for ReLU-based networks.

For each layer \( l \), the weights are sampled from:

$$
W^{[l]} \sim \mathcal{N}\left(0,\; \frac{2}{n^{[l-1]}}\right)
$$


Where:
- n[l-1] is the number of neurons in the previous layer
- Mean = 0
- Variance = 2 / n[l-1]
---

### Intuition Behind the Scaling Factor

ReLU activations zero out approximately **half of their inputs**.
Without compensation, this causes the variance of activations to decrease layer by layer.

The scaling factor is given by:

$$
\sqrt{\frac{2}{n^{[l-1]}}}
$$

increases the weight variance just enough to **preserve the expected magnitude**
of activations across layers, preventing signal decay.

---

### Bias Initialization

All bias vectors are initialized to zero:

$$
b^{[l]} = 0
$$

Biases do not affect variance propagation and initializing them to zero is standard
practice for fully connected networks.

---

### Network Assumptions

This project uses a **4-layer fully connected MLP**:

Input → FC → ReLU → FC → ReLU → FC → ReLU → FC → Output


He Initialization is applied consistently to **all weight matrices** in the network.

---

### Scope of This Stage

`Implemented`:
- He Initialization for all layers
- Correct parameter shapes
- Reproducible random initialization

`Not implemented`:
- Forward propagation
- Loss computation
- Backpropagation
- Training loop

---

### Summary

He Initialization ensures stable activation variance in deep ReLU networks.
This stage establishes a solid numerical foundation for all subsequent steps,
including forward propagation and training.


## Stage 3 — ReLU Activation Function

This stage introduces the **ReLU (Rectified Linear Unit)** activation function
and its derivative, which are used throughout the network.

ReLU is the main non-linearity ensuring the network can model complex relationships.
It is applied element‑wise to all activations in every hidden layer.

---

### ReLU Definition

The ReLU function is mathematically defined as:

$$\text{ReLU}(z) = \max(0, z)$$

It passes positive values unchanged and zeroes out negative values.

In NumPy, this is efficiently implemented using `np.maximum(x, 0)`,  
which operates element‑wise on matrices of any shape.

---

### Derivative of ReLU

The derivative of ReLU with respect to its input is:

$$\frac{\partial \text{ReLU}(z)}{\partial z} = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}$$

In NumPy, it is computed using a boolean mask:

```
(x > 0).astype(int)
```
This derivative is used during **backpropagation** to compute gradients
for all hidden layers.

---

### Why ReLU?

ReLU is chosen because it:

- Introduces non-linearity while remaining computationally efficient
- Helps mitigate the vanishing gradient problem
- Works well with **He Initialization**, used in Stage 2

---

### Scope of This Stage

`Implemented`:
- ReLU activation function
- Derivative of ReLU (dReLU)

`Not implemented`:
- Forward propagation
- Loss computation
- Backpropagation logic


## Stage 4 – Feed Forward Pass

### Overview
In this stage, we implement the **feed‑forward** mechanism of our 4‑layer MLP.  
This part connects all previous stages — initialized parameters and ReLU activation — to compute network outputs for all examples in one vectorized step.

Feed‑forward means each layer takes the activations from the previous layer, applies its affine transformation (weights × inputs + bias), and passes the result through the activation function to produce the next layer’s activations.

---

### Formulas

**Forward propagation for each hidden layer:**

Z[h] = W[h] · A[h‑1] + b[h]  
A[h] = ReLU(Z[h])

Where:
- A[0] = X (input)
- ReLU(z) = max(0, z)
- We use **ReLU** for all layers including the output layer A4 (for non‑negative regression output).

---

### Vectorized Computation — Why It Matters
NumPy performs matrix operations element‑wise in compiled C code.  
Using vectorized syntax (`W @ A + b`) is crucial for speed and correctness — it computes **many dot products in parallel** instead of potentially millions of Python‑level loops.

Conceptually, each entry Z[i,j] is the dot product between:
- the i‑th neuron’s weight vector W[i,:]
- and the j‑th sample’s input vector A[:,j]

Thus matrix multiplication is just **multiple dot products computed simultaneously**.

---

### Scope of This Stage

`Implemented`:
- Full forward pass (4 layers)
- All affine transformations (Z = W@A + b)
- ReLU activation on all layers
- Vectorized NumPy operations
- Cache structure for backprop

`Not implemented`:
- Loss computation (MSE)
- L2 regularization
- Backpropagation
- Training loop


## Stage 5  
### Loss Function (MSE), L2 Regularization & Backpropagation

In this stage, we complete the learning pipeline by defining the **loss function**, adding **L2 regularization**, and implementing **backpropagation** to compute gradients for all parameters.

---

### 1. Mean Squared Error (MSE) Loss

The Mean Squared Error measures the average squared difference between predicted values and true targets.

$$
L = \frac{1}{2m} \sum_{i=1}^{m} (y_d^{(i)} - y_{pred}^{(i)})^2
$$

Where:
- `m` is the number of training examples  
- $y_d$ is the ground truth  
- $y_{pred}$ is the model prediction  

**Why divide by `m`?**  
This makes the loss (and gradients) independent of batch size and stabilizes training.

---

### 2. L2 Regularization

To reduce overfitting and prevent large weights, L2 regularization is added to the loss.

$$
L_{reg} = \frac{1}{m} \frac{\lambda}{2} \sum_{l=1}^{L} \|W^{[l]}\|^2_F
$$

Where:
- $L = 4$ is the number of layers  
- $\lambda$ is the regularization strength  
- $\|W\|_F^2$ is the sum of squared weights  

This term penalizes large weights and encourages simpler models.

---

### 3. Backpropagation (Regression)

Backpropagation computes gradients of the loss with respect to all parameters using the chain rule.

#### 3.1 Gradient at Output Layer

For MSE loss, the gradient with respect to the output activation is:

$$
dA^{[4]} = \frac{\partial L}{\partial A^{[4]}} = \frac{1}{m} (A^{[4]} - Y)
$$

---

#### 3.2 Backward Step for Layer $h$

For each layer (from output to input), gradients are computed as follows:

##### Activation Gradient
$$
dZ^{[h]} = dA^{[h]} * g'^{[h]}(Z^{[h]})
$$

Where:
- $g'$ is the derivative of ReLU (`drelu`)
- `*` denotes element-wise multiplication

---

##### Weight Gradient
$$
dW^{[h]} = dZ^{[h]} \cdot A^{[h-1]T} + \frac{\lambda}{m} W^{[h]}
$$

This consists of:
- The data-driven gradient
- The L2 regularization gradient

---

##### Bias Gradient
$$
db^{[h]} = \sum_{i=1}^{m} dZ^{[h](i)}
$$

(Implemented using `axis=1` and `keepdims=True`)

---

##### Propagating Error to Previous Layer
$$
dA^{[h-1]} = W^{[h]T} \cdot dZ^{[h]}
$$

This step distributes the error back to the previous layer based on connection weights.

---

### 4. Summary

At the end of this stage:
- The model can measure prediction error using MSE  
- Overfitting is controlled via L2 regularization  
- Gradients for all weights and biases are computed using backpropagation  

These components together enable parameter updates during training.


## Current Stage: Stage 6 — Training Loop (Regression)

In this stage, all previously implemented functions are combined to train
the regression model using **Gradient Descent**.

---

###  Training Procedure (Per Iteration)

For each training iteration, the following steps are performed:

---

###  Forward Propagation

Perform a forward pass to compute predictions and cache intermediate values:

$$
(y_{\text{pred}}, \; cache) = \text{feed\_forward}(X, parameters)
$$

---

### Compute MSE Loss

The Mean Squared Error loss is computed as:

$$
loss = \frac{1}{2m} \sum_{i=1}^{m}
\left(y_{\text{pred}}^{(i)} - y_d^{(i)}\right)^2
$$

---

###  Compute L2 Regularization Loss

The L2 regularization term is computed using the network weights:

$$
reg\_loss = \frac{\lambda}{2m}
\sum_{h=1}^{4} \left\| W^{[h]} \right\|_F^2
$$

---

###  Total Loss

The total loss is calculated and stored:

$$
total\_loss = loss + reg\_loss
$$

---

### Backpropagation with Regularization

Gradients of all parameters are computed using:

$$
grads = \text{Backpropagation\_reg}(X, y_d, cache, parameters, \lambda)
$$

---

### Parameter Update (Gradient Descent)

All weights and biases are updated as follows:

$$
W^{[h]} = W^{[h]} - \eta \cdot dW^{[h]}
$$

$$
b^{[h]} = b^{[h]} - \eta \cdot db^{[h]}
$$

---

### Outputs

- `losses`: list containing total loss values over all iterations  
- `parameters`: final trained weights and biases
