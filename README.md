# Scalable Dynamic Degree-Corrected Poisson Matrix Factorization

This repository contains the **Final Year Project (Thesis)** titled "_Scalable Dynamic Degree-Corrected Poisson Matrix Factorisation_", completed at **Imperial College London (2023-2024)**. This work extends the traditional Poisson Matrix Factorization (PMF) model by incorporating **temporal node dynamics** into its structure.

This work provides a **scalable and flexible approach** for modeling temporal dynamics in networked systems, with applications in **link prediction**, **recommender systems**, and **cyber-security**.

## Key Contributions

In this thesis, we introduce a **novel dynamic extension** to the PMF model for **link prediction**, designed to:

- Overcome the **limitations of static models** in evolving networks by incorporating **temporal node dynamics**.
- Provide **efficient model fitting** through a Bayesian hierarchical framework and a **scalable variational inference algorithm**.
- Demonstrate the model's effectiveness through a **Python implementation**, validated on both **synthetic networks** and **real-world datasets**.

## Overview of the Model

The standard PMF model is defined as:

```math
\begin{align*}
    A_{ij}
        & = \mathbb{1}_{ \mathbb{N}_{+} } (N_{ij}),\\
    N_{ij}
        & \sim \text{Poisson} (\mathbf{x}_i^T \mathbf{y}_j) \text{, where } \mathbf{x}_i^T \mathbf{y}_j = \textstyle\sum_{r = 1}^d x_{ir} y_{jr}.
\end{align*}
```

Here:

- $A_{ij}$ is a **binary variable indicating the presence of an interaction** between nodes $i$ and $j$.
- $N_{ij}$ represents the **number of interactions** between nodes $i$ and $j$.
- $\mathbf{x}_i$ and $\mathbf{y}_j$ denote the **latent features** of nodes $i$ and $j$.

However, this formulation is **static**, meaning it cannot capture **changes in network dynamics over time**.

To address this limitation, we propose a **dynamic extension** of the PMF model. Our approach assumes that each node's **latent feature representation remains fixed over time** but is adjusted using **time-dependent degree correction factors**. This allows nodes to **increase or decrease their activity levels over time** while preserving the structure of their interactions.

The mathematical formulation of our model is:

```math
\begin{align*}
  A_{tij}
      & = \mathbb{1}_{ \mathbb{N}_{+} } (N_{tij}),\\
  N_{tij} \mid \mathbf{x}_{i}, \mathbf{y}_{j}, \rho_{ti}^{(x)}, \rho_{tj}^{(y)}
      & \sim \text{Poisson} (\rho_{ti}^{(x)} \rho_{tj}^{(y)} \mathbf{x}_{i}^{\top} \mathbf{y}_{j}),\\
  x_{ir} \mid \zeta_{i}^{(x)}
      & \sim \text{Gamma} (a^{(x)}, \zeta_{i}^{(x)}),\\
  y_{jr} \mid \zeta_{j}^{(y)}
      & \sim \text{Gamma} (a^{(y)}, \zeta_{j}^{(y)}), \\
  \zeta_{i}^{(x)}
      & \sim \text{Gamma} (b^{(x)}, c^{(x)}),\\
  \zeta_{j}^{(y)}
      & \sim \text{Gamma} (b^{(y)}, c^{(y)}),\\
  \rho_{ti}^{(x)}
      & \sim \text{Gamma}_{(0, 1)} (\alpha^{(x)} \beta^{(x)}),\\
  \rho_{tj}^{(y)}
      & \sim \text{Gamma}_{(0, 1)} (\alpha^{(y)}, \beta^{(y)}).
\end{align*}
```

Here:

- $A_{tij}$ is a **binary variable indicating the presence of an interaction** between nodes $i$ and $j$ at time $t$.
- $N_{tij}$ represents the **number of interactions** between nodes $i$ and $j$ at time $t$.
- $\mathbf{x}_i$ and $\mathbf{y}_j$ are the **time-invariant latent features** of nodes $i$ and $j$.
- $\zeta_{i}^{(x)}$ and $\zeta_{j}^{(y)}$ serve as **spread hyperparameters** controlling the distribution of latent features.
- $\rho_{ti}^{(x)}$ and $\rho_{tj}^{(y)}$ are **time-dependent correction factors**, adjusting each node's activity level over time.

To estimate the parameters efficiently, we develop a **scalable variational inference algorithm**. For more details on the model and its implementation, please refer to the **thesis report**.

## Project Structure

The repository is organized as follows:

📂 **`pmf/`** – Implementation of the extended PMF model.  
📂 **`rdpg/`** – Implementation of the RDPG models (AIP, COSIE, DASE, MASE).  
📂 **`report/`** – Code used for generating parts of the final report.  
📂 **`tests/`** – Unit tests to validate the correctness of the algorithms. (Some tests fail due to numerical issues)  
📂 **`utils/`** – Helper functions and utilities.

📜 **`README.md`** – Overview of the project and instructions for usage.  
🚀 **`main.py`** – Main script for running experiments.  
📓 **`main.ipynb`** – Jupyter notebook for interactive model exploration.  
📄 **`requirements.txt`** – List of dependencies required to run the project.  
📑 **`report.pdf`** – Compiled version of the final thesis report.

## Running the Application

To run the application, you need to install the required dependencies and activate the Python environment. Follow these steps:

```bash
# Create a new Python environment
python -m venv env

# Activate the environment
source env/bin/activate

# Install the required packages
pip install -r requirements.txt

# Run the application
python main.py
```

To run the tests, use the following commands.  
**Note**: Some tests may fail due to numerical issues.

```bash
# Run the tests
pytest

# Run tests from a specific directory
pytest path/to/directory

# Run an individual test file
pytest path/to/test_file.py

# Run an individual test
pytest path/to/test_file.py::test_function_name
```
