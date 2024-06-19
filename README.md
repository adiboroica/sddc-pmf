# Scalable Dynamic Degree-Corrected Poisson Matrix Factorization

This repository contains the implementation of an extension of the PMF model that incorporates temporal node dynamics into its structure.

Specifically, the PMF model assumes a fixed latent-space representation of the nodes, static over time, and corrects it with time-dependent scalar degree correction parameters. This allows the nodes to increase or decrease their activity levels over time without changing the type of connections that are formed.

Mathematically, the model is defined as follows:

$$
\begin{align*}
  A_{tij}     
      & = \mathbb{1}\_{ \mathbb{N}\_{+} } (N_{tij}),\\
  N_{tij} \mid \mathbf{x}\_{i}, \mathbf{y}\_{j}, \rho_{ti}^{(x)}, \rho_{tj}^{(y)}
      & \sim \text{Poisson} (\rho_{ti}^{(x)} \rho_{tj}^{(y)} \mathbf{x}\_{i}^{\top} \mathbf{y}\_{j}),\\
  x_{ir} \mid \zeta_{i}^{(x)}
      & \sim \text{Gamma} (a^{(x)}, \zeta_{i}^{(x)}),\\
  y_{jr} \mid \zeta_{j}^{(y)} 
      & \sim \text{Gamma} (a^{(y)}, \zeta_{j}^{(y)}), \\
  \zeta_{i}^{(x)} 
      & \sim \text{Gamma} (b^{(x)}, c^{(x)}),\\
  \zeta_{j}^{(y)} 
      & \sim \text{Gamma} (b^{(y)}, c^{(y)}),\\
  \rho_{ti}^{(x)}  
      & \sim \text{Gamma}\_{(0, 1)} (\alpha^{(x)} \beta^{(x)}),\\
  \rho_{tj}^{(y)}  
      & \sim \text{Gamma}\_{(0, 1)} (\alpha^{(y)}, \beta^{(y)}).
\end{align*}
$$

See the report for more details on the model and the implementation.

## Running the Application

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

## Testing the Application

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
