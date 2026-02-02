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



