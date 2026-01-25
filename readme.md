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


## current stage :Stage 2 — He Initialization

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
