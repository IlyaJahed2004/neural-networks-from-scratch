# Part 3: Analysis Laboratory
## section1: Stochastic Gradient Descent (SGD) & Optimization Analysis

In this phase, we transition from building core architectures to **analyzing model behavior** and implementing advanced optimization techniques. The first major milestone in this analysis is the implementation of **Stochastic Gradient Descent (SGD)** to compare its convergence properties against the standard Batch Gradient Descent used in Part 2.

---

### 1. The Shift from Batch to Stochastic

In Part 2, we used **Batch Gradient Descent**, where the gradient is computed as an average over the **entire dataset** before a single weight update occurs:

$$ W := W - \eta \cdot \frac{1}{m} \sum_{i=1}^{m} \nabla Loss^{(i)} $$

While stable, this approach is computationally expensive for large datasets and can get stuck in local minima (saddle points).

In **Part 3**, we implemented **SGD**, where parameters are updated **for every single training example**:

$$ W := W - \eta \cdot \nabla Loss^{(i)} $$

This results in $m$ updates per epoch (instead of 1), leading to much faster initial learning, albeit with a "noisy" loss curve.

---

### 2. Implementation Details: `training_clf_sgd`

The new training function `training_clf_sgd` introduces a nested loop structure:
1.  **Outer Loop (Epochs):** Controls the number of passes over the dataset.
2.  **Inner Loop (Samples):** Iterates through $m$ examples one by one.

Inside the inner loop, the forward pass, loss computation, and backpropagation are executed on input vectors of shape `(784, 1)` and label vectors of shape `(10, 1)`.

**Key Mathematical Note:**
Even though we process one sample at a time, we retain **L2 Regularization**. The regularization term is added to the loss of each sample to ensure the weights are continuously penalized, preventing overfitting even during stochastic updates.

---

### 3. The Critical Role of Shuffling

A mandatory component of SGD implemented in this stage is **Data Shuffling**.

In many datasets (including MNIST), data might be ordered by class (e.g., all '0's, then all '1's). If we feed data in this order, the gradient will constantly pull the weights in a specific direction for a long time (e.g., "predict 0"), only to be violently pulled in a different direction later. This causes:
*   **Catastrophic Forgetting:** The network forgets what it learned about digit '0' while learning digit '1'.
*   **Non-convergence:** The loss oscillates wildly without settling.

**Our Solution:**
At the start of *every* epoch, we generate a random permutation of indices:
```python
permutation = np.random.permutation(m)
X_shuffled = X[:, permutation]
Y_shuffled = Y[:, permutation]
```
This ensures the assumption of **Independent and Identically Distributed (I.I.D.)** data is respected within every epoch, smoothing out the trajectory towards the global minimum.


### 4. Expected Behavior (Analysis)

By comparing the two models (Batch vs. SGD) in the upcoming analysis, we expect to observe:
*   **Batch GD:** Smooth, monotonic decrease in loss, but slow convergence (requires many epochs).
*   **SGD:** rapid drop in loss initially, followed by fluctuations (noise) around the minimum. The noise acts as a regularizer, potentially helping the model escape sharp local minima.



## Section 2: Activation Function Analysis (ReLU vs. Sigmoid)

After exploring optimization strategies with SGD, the second pillar of our analysis focuses on the **internal dynamics** of the network. Specifically, we investigate the impact of the hidden layer activation functions on convergence speed and training stability.

In Part 1 and 2, we exclusively used **ReLU (Rectified Linear Unit)**. In this section, we implement **Sigmoid** to empirically demonstrate the **Vanishing Gradient Problem**, a fundamental concept in Deep Learning theory.

---

### 1. Developing a Flexible Architecture

To perform a valid comparative analysis, we refactored the hard-coded architecture into a **flexible framework**. This adheres to the **DRY (Don't Repeat Yourself)** software engineering principle.

Instead of writing separate training functions for each activation type, we implemented:
1.  **`feed_forward_flexible`**: Accepts a `hidden_activation` argument ('relu' or 'sigmoid'). It dynamically selects the activation function handle (`g`) for layers 1, 2, and 3. *Note: Layer 4 remains fixed as Softmax for classification.*
2.  **`Backpropagation_flexible`**: Similarly selects the corresponding derivative function handle (`dg`) to ensure the chain rule is applied correctly.

This setup ensures that **only** the activation function changes between experiments, while initialization, architecture dimensions, and data processing remain constant controls.

---

### 2. Sigmoid Implementation & Numerical Stability

We introduced the Sigmoid function and its derivative:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
$$ \sigma'(z) = \sigma(z) \cdot (1 - \sigma(z)) $$

**Implementation Note:**
A naive implementation of Sigmoid can lead to numerical overflow when $z$ is a large negative number (causing $e^{-z} \to \infty$). To ensure **Numerical Stability**, we use NumPy's stable implementation or clamp values where necessary, ensuring the training process remains robust even with unnormalized weights.

---

### 3. The Vanishing Gradient Problem (Theoretical Analysis)

The primary hypothesis of this experiment is that the **ReLU** network will converge significantly faster than the **Sigmoid** network. This is due to the **Vanishing Gradient Problem**:

*   **Sigmoid Saturation:** The maximum value of the sigmoid derivative is $0.25$ (at $z=0$). As $|z|$ increases, the derivative approaches $0$.
*   **Chain Rule Multiplication:** In a deep network (4 layers), the gradients are computed via the chain rule. If we multiply small numbers (e.g., $< 0.25$) repeatedly across layers:
    $$ \frac{\partial L}{\partial W^{[1]}} \propto \sigma'(z^{[4]}) \cdot \sigma'(z^{[3]}) \cdot \sigma'(z^{[2]}) \dots $$
    The gradient exponentially decays towards zero.
*   **Result:** The weights in the early layers ($W1, W2$) receive tiny updates, effectively stopping the "learning" process for those layers.

**ReLU Advantage:**
In contrast, the derivative of ReLU is either $1$ (for $z>0$) or $0$. Multiplying by $1$ preserves the gradient magnitude through backpropagation, allowing deep networks to learn efficiently.

### 4. Experimental Setup

To verify this, we run two parallel training sessions:
1.  **Model A (Baseline):** Standard ReLU hidden layers.
2.  **Model B (Experiment):** Sigmoid hidden layers.

Both models share the **exact same random seed** for weight initialization. This guarantees that any difference in the Loss Curve is solely attributed to the activation function mechanics, confirming the theoretical predictions.


## Section 3: Hyperparameter Tuning - Learning Rate Analysis

The final phase of our analysis investigates the most critical hyperparameter in deep learning: the **Learning Rate ($\eta$)**. This parameter controls the step size taken towards the minimum of the loss function during each update:

$$ W := W - \eta \cdot \nabla Loss $$

If $\eta$ is ill-chosen, the model may fail to train entirely, regardless of the architecture's quality.

---

### 1. Experimental Setup

To observe the effects of step size on convergence, we conduct three controlled experiments using the **Batch Gradient Descent** strategy on the full MNIST training set for a fixed duration of **100 iterations**.

**Scenarios:**
1.  **High Learning Rate ($\eta = 1.0$):** Aggressive updates.
2.  **Baseline Learning Rate ($\eta = 0.1$):** Standard updates.
3.  **Low Learning Rate ($\eta = 0.001$):** Conservative updates.

*Note: All other hyperparameters (initialization seed, architecture, activation=ReLU) remain constant.*

---

### 2. Theoretical Expectations & Analysis

#### Case A: High Learning Rate ($\eta = 1.0$) - "Overshooting"
*   **Observation:** We expect the loss to oscillate effectively or even increase (diverge).
*   **Theory:** When the step size is too large, the algorithm steps **over** the minimum. The gradient at the new position might point back with even greater magnitude, causing the weights to bounce back and forth, moving further away from the optimal valley. This confirms the **"Overshooting"** phenomenon discussed in lectures.

#### Case B: Low Learning Rate ($\eta = 0.001$) - "Slow Convergence"
*   **Observation:** The loss decreases monotonically but extremely slowly.
*   **Theory:** The updates are tiny. While safe (unlikely to overshoot), the model requires orders of magnitude more iterations to reach the same performance as the baseline. In a resource-constrained environment, this is computationally inefficient and may get stuck in shallow local plateaus.

#### Case C: Low Learning Rate ($\eta = 0.001$) - "Slow Convergence"
*   **Observation:** The loss decreases monotonically, but the slope is extremely flat. After 100 iterations, the loss is significantly higher than the baseline ($\eta=0.1$).
*   **Theory:** The step size is too small. The model is "learning," but at a snail's pace. To reach the global minimum with this rate, we would need thousands of iterations, making it computationally inefficient. It is also more prone to getting stuck in local minima or saddle points because it lacks the momentum to escape them.

### 4. Conclusion

This experiment confirms that the Learning Rate is a trade-off parameter:
1.  **Too High:** Causes instability and divergence (Overshooting).
2.  **Too Low:** Results in safe but painfully slow convergence.
3.  **Optimal:** A value (like 0.1 in this specific landscape) that allows for rapid initial descent without overshooting the target.

**Key Takeaway:** Tuning $\eta$ is often the first step in hyperparameter optimization. Advanced optimizers (like Adam or RMSProp, not implemented here) attempt to solve this by adapting the learning rate automatically during training.

## Section 4: Generalization & L2 Regularization Analysis

The final experiment addresses one of the most common challenges in Machine Learning: **Overfitting**.

### 1. The Concept of Overfitting
Overfitting occurs when a neural network learns the training data *too well*, including its noise and outliers. This results in:
*   **High Training Accuracy:** The model memorizes the training set.
*   **Low Test Accuracy:** The model fails to generalize to new, unseen data.

To combat this, we utilize **L2 Regularization** (Weight Decay). By adding a penalty term to the loss function ($\frac{\lambda}{2m} \sum \|W\|^2$), we force the weights to remain small. Smaller weights usually lead to smoother decision boundaries and better generalization.

---

### 2. Experimental Setup

We train two identical models (same architecture, same initialization, same learning rate $\eta=0.1$) for **1000 iterations**, varying only the regularization parameter $\lambda$:

1.  **Model A (Unregularized):** $\lambda = 0$. This model is free to grow its weights indefinitely to minimize the training loss.
2.  **Model B (Regularized):** $\lambda = 0.1$. This model penalizes large weights.

**Metrics:**
We evaluate performance not just by Loss, but by **Accuracy** on both the **Training Set** (memorization) and the **Test Set** (generalization).

---

### 4. Analysis

Based on the results generated above:

1.  **High Variance (Overfitting):** Model A ($\lambda=0$) typically achieves a slightly higher **Train Accuracy** but a lower **Test Accuracy**. The "Gap" between train and test performance is larger, indicating the model has overfitted to the training noise.
2.  **Generalization:** Model B ($\lambda=0.1$) might have a slightly lower Train Accuracy (because the penalty constrains it), but it should achieve a **higher Test Accuracy**. The gap is smaller, proving that L2 Regularization successfully forced the model to learn more robust features rather than memorizing specific examples.

**Why does L2 help?**
By punishing large weights ($W$), the network is discouraged from relying heavily on any single feature (pixel). This distributes the "reasoning" across many neurons, making the prediction less sensitive to small variations in input, which is the definition of robustness.



## Section 4: The Impact of Weight Initialization

Weight initialization is often overlooked, but it is a cornerstone of deep learning dynamics. In this experiment, we compare standard **He Initialization** against a naive **Small Random Initialization**.

### 1. Experimental Setup
*   **Model A (He Init):** Weights sampled from $\mathcal{N}(0, \frac{2}{n_{in}})$. This keeps the variance of activations constant across layers.
*   **Model B (Bad Init):** Weights sampled from $\mathcal{N}(0, 1) \times 0.01$.
*   **Goal:** To verify if the network can learn when starting with extremely small weights.

### 2. Observation
The loss curve comparison reveals a stark difference:
*   **He Initialization:** The loss drops rapidly, indicating successful learning and feature extraction.
*   **Bad Initialization:** The loss remains nearly constant (flatline). The model fails to improve beyond its initial random state, likely yielding an accuracy around 10% (random guessing for 10 classes).

### 3. Theoretical Analysis: Why did it fail?

The failure of the "Bad Init" model is caused by the **Vanishing Gradient Problem**, which occurs in two phases:

#### A. Forward Pass (Signal Decay)
In a deep network, activations are computed as $A^{[l]} = \text{ReLU}(W^{[l]}A^{[l-1]} + b)$.
Since initialized weights are tiny ($W \approx 0.01$), the output of each layer becomes progressively smaller. By the time the signal reaches the output layer ($A^{[4]}$), the values are effectively zero.

#### B. Backward Pass (Gradient Vanishing)
The gradient update for the first layer ($W1$) depends on the product of weights from later layers via the Chain Rule:
$$ \frac{\partial L}{\partial W^{[1]}} \propto W^{[4]} \cdot W^{[3]} \cdot W^{[2]} \dots $$

When we multiply these tiny matrices ($0.01 \times 0.01 \times \dots$) during backpropagation, the resulting gradient becomes infinitesimally small (approaching machine epsilon).
$$ W_{new} = W_{old} - \eta \cdot (\approx 0) $$
As a result, the weights do not update, and the network remains "stuck" at its initialization point.

### 4. Conclusion
This confirms that **He Initialization** is essential for ReLU networks. It ensures that gradients flow efficiently through the network, preventing both the Vanishing and Exploding gradient problems.
