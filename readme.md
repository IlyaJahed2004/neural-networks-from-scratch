# Neural Networks From Scratch (NumPy)

This repository contains an incremental implementation of a Multi-Layer Perceptron (MLP)
built entirely from scratch using NumPy. The project emphasizes a clear, modular
development process, with each core component of the neural network being implemented
step-by-step and tracked through isolated commits.

## Current Stage: Data Preprocessing Pipeline

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

### Next Steps:
*   Model architecture definition, parameter initialization (He Initialization),
    and activation functions (ReLU).
*   Implementation of the forward propagation pass.
*   Development of loss functions (Mean Squared Error) and regularization.
*   Implementation of the backward propagation algorithm.
*   Integration into a comprehensive training loop.

## Project Philosophy

Each core component of the neural network is implemented and committed in a focused manner,
allowing for a clean Git history that reflects the incremental development of a complex system.
This approach facilitates easier review, debugging, and understanding of the project's evolution.
